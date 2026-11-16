import AppKit

/// Descarga guiada de modelos (primer arranque). Es el único momento en que la app usa internet.
/// Corre `python -m runtime.setup` (revisiones fijas de los pesos, Marian convertido a int8) con red habilitada;
/// el motor de traducción en sí sigue arrancando con HF_HUB_OFFLINE=1. Los pesos son datos, no código.
/// Se guardan en el contenedor del sandbox (Application Support/OpenTraductor).
final class ModelSetup: NSObject {
    /// Tamaño aproximado de todo lo que baja el setup (medido: 3,24 GB con Parakeet, Kokoro, Marian fuente y convertido).
    static let expectedBytes: Int64 = 3_250_000_000
    /// Espacio libre mínimo recomendado: descarga + conversión + margen.
    static let requiredFreeBytes: Int64 = 5_000_000_000

    // MARK: Comprobación (sin red)

    /// `python -m runtime check` en modo offline: código 0 = todo en disco.
    static func modelsReady(completion: @escaping (Bool) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            let p = Process()
            p.executableURL = RuntimeProcess.pythonURL()
            p.arguments = ["-m", "runtime", "check"]
            p.currentDirectoryURL = RuntimeProcess.appDirURL()
            var env = (try? RuntimeProcess.makeEnvironment()) ?? ProcessInfo.processInfo.environment
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            p.environment = env
            p.standardOutput = FileHandle.nullDevice
            p.standardError = FileHandle.nullDevice
            var ok = false
            do { try p.run(); p.waitUntilExit(); ok = p.terminationStatus == 0 } catch { ok = false }
            DispatchQueue.main.async { completion(ok) }
        }
    }

    // MARK: Ventana

    private let parent: NSWindow
    private let onReady: () -> Void
    private let panel: NSWindow
    private let titleLabel = NSTextField(labelWithString: "")
    private let bodyLabel = NSTextField(wrappingLabelWithString: "")
    private let statusLabel = NSTextField(labelWithString: "")
    private let detailLabel = NSTextField(wrappingLabelWithString: "")
    private let bar = NSProgressIndicator()
    private let primary = NSButton(title: "", target: nil, action: nil)
    private let secondary = NSButton(title: "", target: nil, action: nil)

    private var process: Process?
    private var timer: Timer?
    private var cancelled = false
    private var stderrTail = ""
    private var stage = ""

    init(parent: NSWindow, onReady: @escaping () -> Void) {
        self.parent = parent
        self.onReady = onReady
        panel = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 520, height: 300),
                         styleMask: [.titled], backing: .buffered, defer: false)
        super.init()
        buildUI()
    }

    func present() {
        showIntro()
        parent.beginSheet(panel)
    }

    private func buildUI() {
        titleLabel.font = .systemFont(ofSize: 17, weight: .semibold)
        bodyLabel.font = .systemFont(ofSize: 13)
        statusLabel.font = .systemFont(ofSize: 12)
        statusLabel.textColor = .secondaryLabelColor
        detailLabel.font = .systemFont(ofSize: 11)
        detailLabel.textColor = .secondaryLabelColor
        detailLabel.maximumNumberOfLines = 4
        bar.isIndeterminate = false
        bar.minValue = 0
        bar.maxValue = 1
        bar.style = .bar
        primary.bezelStyle = .rounded
        primary.keyEquivalent = "\r"
        primary.target = self
        primary.action = #selector(primaryTapped)
        secondary.bezelStyle = .rounded
        secondary.target = self
        secondary.action = #selector(secondaryTapped)

        let buttons = NSStackView(views: [secondary, primary])
        buttons.orientation = .horizontal
        buttons.spacing = 12
        let stack = NSStackView(views: [titleLabel, bodyLabel, bar, statusLabel, detailLabel, buttons])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 12
        stack.edgeInsets = NSEdgeInsets(top: 24, left: 24, bottom: 20, right: 24)
        stack.translatesAutoresizingMaskIntoConstraints = false
        let content = NSView()
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            stack.topAnchor.constraint(equalTo: content.topAnchor),
            stack.bottomAnchor.constraint(equalTo: content.bottomAnchor),
            content.widthAnchor.constraint(equalToConstant: 520),
            bar.widthAnchor.constraint(equalToConstant: 472),
            bodyLabel.widthAnchor.constraint(equalToConstant: 472),
            detailLabel.widthAnchor.constraint(equalToConstant: 472),
        ])
        buttons.setHuggingPriority(.required, for: .vertical)
        panel.contentView = content
    }

    // MARK: Estados

    private enum State { case intro, downloading, failed }
    private var state: State = .intro

    private func showIntro() {
        state = .intro
        let free = Self.freeBytes()
        titleLabel.stringValue = "Descarga inicial de modelos"
        bodyLabel.stringValue = """
        opentraductor traduce tu voz sin salir de tu Mac. Para eso necesita bajar una sola vez los modelos de voz y traducción \
        (unos 3,2 GB) desde Hugging Face. Después funciona sin internet. No se envía audio ni ningún dato tuyo.
        """
        bar.doubleValue = 0
        bar.isHidden = true
        statusLabel.stringValue = free == Int64.max ? "" : "Espacio libre en este Mac: \(Self.format(free))."
        detailLabel.stringValue = free < Self.requiredFreeBytes
            ? "Hace falta al menos \(Self.format(Self.requiredFreeBytes)) libres. Liberá espacio y reintentá."
            : ""
        primary.title = "Descargar modelos"
        primary.isEnabled = free >= Self.requiredFreeBytes
        secondary.title = "Salir"
        secondary.isHidden = false
    }

    private func showDownloading() {
        state = .downloading
        titleLabel.stringValue = "Descargando modelos…"
        bar.isHidden = false
        bar.isIndeterminate = false
        bar.doubleValue = 0
        statusLabel.stringValue = "Preparando…"
        detailLabel.stringValue = "Podés cancelar y retomar después: lo ya descargado se conserva."
        primary.title = "Cancelar"
        primary.isEnabled = true
        secondary.isHidden = true
    }

    private func showFailure(_ message: String, detail: String) {
        state = .failed
        timer?.invalidate(); timer = nil
        bar.isHidden = true
        titleLabel.stringValue = "No se pudo completar la descarga"
        statusLabel.stringValue = message
        detailLabel.stringValue = detail
        primary.title = "Reintentar"
        primary.isEnabled = true
        secondary.title = "Salir"
        secondary.isHidden = false
    }

    // MARK: Acciones

    @objc private func primaryTapped() {
        switch state {
        case .intro, .failed: startDownload()
        case .downloading: cancelDownload()
        }
    }

    @objc private func secondaryTapped() { NSApp.terminate(nil) }

    private func cancelDownload() {
        cancelled = true
        process?.terminate()
    }

    private func startDownload() {
        cancelled = false
        stderrTail = ""
        stage = ""
        showDownloading()
        let p = Process()
        p.executableURL = RuntimeProcess.pythonURL()
        p.arguments = ["-u", "-m", "runtime.setup"]
        p.currentDirectoryURL = RuntimeProcess.appDirURL()
        do {
            var env = try RuntimeProcess.makeEnvironment()
            env.removeValue(forKey: "HF_HUB_OFFLINE")      // único paso con red
            env.removeValue(forKey: "TRANSFORMERS_OFFLINE")
            env["HF_HUB_DISABLE_TELEMETRY"] = "1"
            env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            env["DO_NOT_TRACK"] = "1"
            // Con Xet el archivo se arma en memoria y aparece completo al final: la barra no avanzaría.
            // Con HTTP clásico el .incomplete crece en disco y el progreso es real.
            env["HF_HUB_DISABLE_XET"] = "1"
            p.environment = env
        } catch {
            return showFailure("No se pudo preparar la descarga.", detail: error.localizedDescription)
        }
        let out = Pipe(), err = Pipe()
        p.standardOutput = out
        p.standardError = err
        out.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            DispatchQueue.main.async { self?.handleOutput(text) }
        }
        err.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            DispatchQueue.main.async { guard let self else { return }; self.stderrTail = String((self.stderrTail + text).suffix(1500)) }
        }
        p.terminationHandler = { [weak self] proc in
            out.fileHandleForReading.readabilityHandler = nil
            err.fileHandleForReading.readabilityHandler = nil
            DispatchQueue.main.async { self?.finished(status: proc.terminationStatus) }
        }
        do { try p.run() } catch {
            return showFailure("No se pudo iniciar la descarga.", detail: error.localizedDescription)
        }
        process = p
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 0.7, repeats: true) { [weak self] _ in self?.pollProgress() }
    }

    private func handleOutput(_ text: String) {
        for line in text.split(separator: "\n") {
            if line.hasPrefix("ASR") { stage = "Reconocimiento de voz" }
            else if line.hasPrefix("TTS") { stage = "Voz en inglés" }
            else if line.hasPrefix("MT") && line.contains("convirtiendo") {
                stage = "Optimizando el traductor para tu Mac"
                bar.isIndeterminate = true
                bar.startAnimation(nil)
                statusLabel.stringValue = stage + "…"
            }
            else if line.hasPrefix("MT") { stage = "Traductor" }
        }
    }

    private func pollProgress() {
        guard state == .downloading, !bar.isIndeterminate else { return }
        let support = RuntimeProcess.supportDir()
        let dirs = [support.appendingPathComponent("hf"), support.appendingPathComponent("models")]
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let bytes = Self.directorySize(dirs)
            DispatchQueue.main.async {
                guard let self, self.state == .downloading, !self.bar.isIndeterminate else { return }
                let frac = min(0.99, Double(bytes) / Double(Self.expectedBytes))
                self.bar.doubleValue = frac
                let label = self.stage.isEmpty ? "Descargando" : "Descargando: \(self.stage)"
                self.statusLabel.stringValue = "\(label) · \(Self.format(min(bytes, Self.expectedBytes))) de \(Self.format(Self.expectedBytes))"
            }
        }
    }

    private func finished(status: Int32) {
        process = nil
        timer?.invalidate(); timer = nil
        if cancelled { return showIntro() }
        if status == 0 {
            parent.endSheet(panel)
            panel.orderOut(nil)
            return onReady()
        }
        let tail = stderrTail.trimmingCharacters(in: .whitespacesAndNewlines)
        showFailure("Revisá tu conexión a internet y el espacio libre en disco, y volvé a intentar. Lo ya descargado se conserva.",
                    detail: String(tail.suffix(400)))
    }

    // MARK: Utilidades

    static func format(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }

    static func freeBytes() -> Int64 {
        let url = RuntimeProcess.supportDir()
        let probe = FileManager.default.fileExists(atPath: url.path) ? url : url.deletingLastPathComponent()
        let v = try? probe.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return v?.volumeAvailableCapacityForImportantUsage ?? Int64.max
    }

    /// Suma archivos regulares (no enlaces simbólicos: la caché de Hugging Face enlaza snapshots a blobs).
    static func directorySize(_ roots: [URL]) -> Int64 {
        var total: Int64 = 0
        let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey]
        for root in roots {
            guard let e = FileManager.default.enumerator(at: root, includingPropertiesForKeys: keys, options: []) else { continue }
            for case let f as URL in e {
                guard let v = try? f.resourceValues(forKeys: Set(keys)), v.isRegularFile == true else { continue }
                total += Int64(v.fileSize ?? 0)
            }
        }
        return total
    }
}
