import Foundation
import CryptoKit

struct DownloadProgress {
    var doneBytes: Int64
    var totalBytes: Int64
    var bytesPerSecond: Double
    var fileLabel: String
    var fileIndex: Int  // desde 1
    var fileCount: Int

    var fraction: Double { totalBytes > 0 ? min(1, Double(doneBytes) / Double(totalBytes)) : 0 }
    var remainingSeconds: Double? {
        bytesPerSecond > 50_000 ? Double(max(0, totalBytes - doneBytes)) / bytesPerSecond : nil
    }
}

enum DownloadEvent {
    case progress(DownloadProgress)
    case verifying(String)
    case retrying(String)
    case paused
    case finished
    case failed(String)
}

/// Descarga archivos con URLSession a `.part`, de a uno. Reanuda con `Range` desde lo que haya en disco (también
/// después de cerrar la app), reintenta cuando se corta la red y verifica el checksum antes de dejar el archivo en su lugar.
/// Todo el estado vive en `queue` (serie); los eventos se entregan en el hilo principal.
final class ModelDownloader: NSObject, URLSessionDataDelegate {
    private enum Abort { case http(Int), disk(String), restart }

    private let files: [ModelFile]
    private let hub: URL
    private let emit: (DownloadEvent) -> Void
    private let queue = OperationQueue()
    private var session: URLSession!

    private let totalBytes: Int64
    private var index = 0
    private var completedBytes: Int64 = 0
    private var currentBytes: Int64 = 0
    private var handle: FileHandle?
    private var partURL: URL?
    private var task: URLSessionDataTask?
    private var abort: Abort?
    private var running = false
    private var userPaused = false
    private var epoch = 0
    private var attempt = 0
    private var freshRetryUsed = false
    private var lastEmit = Date.distantPast
    private var samples: [(Date, Int64)] = []
    private static let maxAttempts = 8

    init(files: [ModelFile], hub: URL, emit: @escaping (DownloadEvent) -> Void) {
        self.files = files
        self.hub = hub
        self.emit = emit
        self.totalBytes = files.reduce(0) { $0 + $1.size }
        super.init()
        queue.maxConcurrentOperationCount = 1
        queue.name = "opentraductor.modeldownloader"
        let cfg = URLSessionConfiguration.default
        cfg.waitsForConnectivity = true
        cfg.timeoutIntervalForRequest = 60
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        cfg.urlCache = nil
        cfg.httpCookieStorage = nil
        session = URLSession(configuration: cfg, delegate: self, delegateQueue: queue)
    }

    // MARK: Control

    func start() {
        queue.addOperation { [self] in
            guard !running else { return }
            running = true
            userPaused = false
            epoch += 1
            index = 0
            completedBytes = 0
            currentBytes = 0
            attempt = 0
            samples = []
            beginNextFile()
        }
    }

    /// Corta la conexión y deja el `.part` en disco: `start()` retoma desde ahí.
    func pause() {
        queue.addOperation { [self] in
            guard running else { return }
            userPaused = true
            epoch += 1
            if let t = task { t.cancel() } else {
                running = false
                post(.paused)
            }
        }
    }

    func shutdown() {
        queue.addOperation { [self] in
            try? handle?.close()
            handle = nil
            session.invalidateAndCancel()
        }
    }

    // MARK: Archivos

    private func beginNextFile() {
        let fm = FileManager.default
        while index < files.count {
            let f = files[index]
            let dest = f.destination(hub: hub)
            try? fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            if fm.fileExists(atPath: dest.path) {
                if size(of: dest) == f.size {
                    post(.verifying(f.label))
                    if verify(dest, f) {
                        completedBytes += f.size
                        index += 1
                        continue
                    }
                }
                try? fm.removeItem(at: dest)
            }
            return startRequest(f, dest)
        }
        running = false
        post(.finished)
    }

    private func startRequest(_ f: ModelFile, _ dest: URL) {
        let fm = FileManager.default
        let part = dest.appendingPathExtension("part")
        partURL = part
        if !fm.fileExists(atPath: part.path) { fm.createFile(atPath: part.path, contents: nil) }
        var offset = size(of: part)
        if offset > f.size {
            try? fm.removeItem(at: part)
            fm.createFile(atPath: part.path, contents: nil)
            offset = 0
        }
        currentBytes = offset
        if offset == f.size { return finishFile(f, dest) }
        do {
            let h = try FileHandle(forWritingTo: part)
            try h.seekToEnd()
            handle = h
        } catch {
            return fail("No se pudo escribir en el disco: \(error.localizedDescription)")
        }
        var req = URLRequest(url: f.url)
        req.cachePolicy = .reloadIgnoringLocalCacheData
        if offset > 0 { req.setValue("bytes=\(offset)-", forHTTPHeaderField: "Range") }
        abort = nil
        let t = session.dataTask(with: req)
        task = t
        t.resume()
        emitProgress(force: true)
    }

    private func finishFile(_ f: ModelFile, _ dest: URL) {
        let fm = FileManager.default
        guard let part = partURL else { return fail("Error interno de la descarga.") }
        post(.verifying(f.label))
        if size(of: part) == f.size, verify(part, f) {
            try? fm.removeItem(at: dest)
            do { try fm.moveItem(at: part, to: dest) } catch {
                return fail("No se pudo guardar \(f.path): \(error.localizedDescription)")
            }
            completedBytes += f.size
            currentBytes = 0
            index += 1
            attempt = 0
            freshRetryUsed = false
            return beginNextFile()
        }
        try? fm.removeItem(at: part)
        if !freshRetryUsed {
            freshRetryUsed = true
            post(.retrying("Un archivo llegó dañado. Se vuelve a bajar."))
            startRequest(f, dest)
        } else {
            fail("La verificación de «\(f.label)» falló dos veces. Probá de nuevo más tarde o con otra red.")
        }
    }

    private func fail(_ message: String) {
        try? handle?.close()
        handle = nil
        running = false
        post(.failed(message))
    }

    // MARK: URLSessionDataDelegate (corre en `queue`)

    func urlSession(_ session: URLSession, taskIsWaitingForConnectivity task: URLSessionTask) {
        post(.retrying("Sin conexión a internet. La descarga sigue cuando vuelva."))
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        guard let http = response as? HTTPURLResponse else {
            abort = .http(0)
            return completionHandler(.cancel)
        }
        switch http.statusCode {
        case 206:
            completionHandler(.allow)
        case 200:
            // El servidor ignoró el pedido de rango: se empieza de cero.
            if currentBytes > 0 {
                try? handle?.truncate(atOffset: 0)
                try? handle?.seek(toOffset: 0)
                currentBytes = 0
            }
            completionHandler(.allow)
        case 416:
            abort = .restart
            completionHandler(.cancel)
        default:
            abort = .http(http.statusCode)
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        do { try handle?.write(contentsOf: data) } catch {
            abort = .disk(error.localizedDescription)
            dataTask.cancel()
            return
        }
        currentBytes += Int64(data.count)
        attempt = 0  // mientras lleguen datos, la conexión sirve
        emitProgress()
    }

    func urlSession(_ session: URLSession, task finished: URLSessionTask, didCompleteWithError error: Error?) {
        guard finished === task else { return }
        task = nil
        try? handle?.close()
        handle = nil
        guard index < files.count else { return }
        let f = files[index]
        let dest = f.destination(hub: hub)
        if let reason = abort {
            abort = nil
            switch reason {
            case .restart:
                if let part = partURL { try? FileManager.default.removeItem(at: part) }
                return startRequest(f, dest)
            case .disk(let m):
                return fail("No se pudo escribir en el disco: \(m)")
            case .http(let code):
                if code >= 500 || code == 429 || code == 0 { return retry(f, dest, "El servidor está ocupado.") }
                return fail("El servidor respondió con el código \(code) al pedir «\(f.label)».")
            }
        }
        if userPaused {
            running = false
            return post(.paused)
        }
        if let error {
            let code = (error as NSError).code
            if code == NSURLErrorCancelled { return }
            return retry(f, dest, "Se cortó la conexión.")
        }
        finishFile(f, dest)
    }

    private func retry(_ f: ModelFile, _ dest: URL, _ why: String) {
        attempt += 1
        if attempt > Self.maxAttempts {
            return fail("No se pudo seguir bajando «\(f.label)». Revisá tu conexión a internet y reintentá: lo ya descargado se conserva.")
        }
        post(.retrying("\(why) Reintentando (\(attempt) de \(Self.maxAttempts))…"))
        let e = epoch
        let delay = min(30.0, 2.0 * Double(attempt))
        DispatchQueue.global().asyncAfter(deadline: .now() + delay) { [weak self] in
            self?.queue.addOperation {
                guard let self, self.epoch == e, self.running else { return }
                self.startRequest(f, dest)
            }
        }
    }

    // MARK: Progreso

    private func emitProgress(force: Bool = false) {
        let now = Date()
        guard force || now.timeIntervalSince(lastEmit) >= 0.25 else { return }
        lastEmit = now
        let done = completedBytes + currentBytes
        samples.append((now, done))
        samples.removeAll { now.timeIntervalSince($0.0) > 6 }
        var speed = 0.0
        if let first = samples.first, let last = samples.last {
            let dt = last.0.timeIntervalSince(first.0)
            if dt > 0.5 { speed = Double(last.1 - first.1) / dt }
        }
        let label = index < files.count ? files[index].label : ""
        post(.progress(DownloadProgress(doneBytes: done, totalBytes: totalBytes, bytesPerSecond: max(0, speed),
                                        fileLabel: label, fileIndex: min(index + 1, files.count), fileCount: files.count)))
    }

    private func post(_ event: DownloadEvent) {
        let deliver = emit
        DispatchQueue.main.async { deliver(event) }
    }

    // MARK: Verificación

    private func size(of url: URL) -> Int64 {
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs?[.size] as? NSNumber)?.int64Value ?? 0
    }

    private func verify(_ url: URL, _ f: ModelFile) -> Bool {
        switch f.checksum {
        case .sha256(let expected):
            return hexDigest(of: url, using: SHA256()) == expected
        case .gitBlobSHA1(let expected):
            return hexDigest(of: url, using: Insecure.SHA1(), prefix: Data("blob \(f.size)\0".utf8)) == expected
        }
    }

    private func hexDigest<H: HashFunction>(of url: URL, using initial: H, prefix: Data = Data()) -> String? {
        guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? h.close() }
        var hasher = initial
        if !prefix.isEmpty { hasher.update(data: prefix) }
        var reading = true
        while reading {
            autoreleasepool {
                if let chunk = try? h.read(upToCount: 4 << 20), !chunk.isEmpty {
                    hasher.update(data: chunk)
                } else {
                    reading = false
                }
            }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
