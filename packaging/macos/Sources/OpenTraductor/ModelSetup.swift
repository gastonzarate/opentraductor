import AppKit
import AVFoundation
import SwiftUI

/// Arranque guiado del primer uso: 1) permiso de micrófono, 2) descarga de modelos con URLSession (ModelDownloader),
/// 3) preparación del traductor y aviso para arrancar el motor. Es el único momento en que la app usa internet.
/// El motor Python sigue arrancando con HF_HUB_OFFLINE=1. Los pesos son datos, no código, y se guardan en el
/// contenedor del sandbox (Application Support/OpenTraductor).
final class ModelSetup: NSObject {
    private let parent: NSWindow
    private let panel: NSWindow
    private let model: SetupModel

    init(parent: NSWindow, onReady: @escaping () -> Void) {
        self.parent = parent
        let m = SetupModel()
        self.model = m
        self.panel = NSWindow(contentViewController: NSHostingController(rootView: SetupView(model: m)))
        super.init()
        panel.styleMask = [.titled]
        m.onQuit = { NSApp.terminate(nil) }
        m.onReady = { [weak self] in
            guard let self else { return }
            self.parent.endSheet(self.panel)
            self.panel.orderOut(nil)
            onReady()
        }
    }

    func present() { parent.beginSheet(panel) }

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

    /// Si los modelos ya estaban y el micrófono nunca se consultó, pide el permiso antes de arrancar el motor.
    static func requestMicrophoneIfNeeded(completion: @escaping () -> Void) {
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .notDetermined else { return completion() }
        AVCaptureDevice.requestAccess(for: .audio) { _ in DispatchQueue.main.async { completion() } }
    }

    // MARK: Preparación del traductor

    /// Deja el traductor (Marian CTranslate2 int8) listo. Con la copia ya convertida dentro del bundle no hay nada que hacer
    /// (el motor la lee de Resources con OPENTRADUCTOR_MARIAN_CT2); si no, se convierte con el Python embebido (sin red).
    /// Devuelve nil si salió bien, o el motivo del fallo.
    static func prepareTranslator() -> String? {
        let fm = FileManager.default
        let dest = ModelManifest.installedMarianDir()
        let required = ["model.bin", "source.spm", "target.spm"]
        if ModelManifest.bundledMarianDir() != nil { return nil }
        if required.allSatisfy({ fm.fileExists(atPath: dest.appendingPathComponent($0).path) }) { return nil }
        // Sin copia en el bundle: conversión local con el Python embebido, sin red.
        let p = Process()
        p.executableURL = RuntimeProcess.pythonURL()
        p.arguments = ["-c", "from runtime.setup import convert_marian; convert_marian()"]
        p.currentDirectoryURL = RuntimeProcess.appDirURL()
        do {
            var env = try RuntimeProcess.makeEnvironment()
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            p.environment = env
            let err = Pipe()
            p.standardOutput = FileHandle.nullDevice
            p.standardError = err
            try p.run()
            let data = err.fileHandleForReading.readDataToEndOfFile()
            p.waitUntilExit()
            if p.terminationStatus != 0 {
                let tail = String(String(data: data, encoding: .utf8)?.suffix(400) ?? "")
                return "No se pudo preparar el traductor (código \(p.terminationStatus)). \(tail)"
            }
        } catch {
            return "No se pudo preparar el traductor: \(error.localizedDescription)"
        }
        return nil
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
}

// MARK: - Modelo de la pantalla

final class SetupModel: ObservableObject {
    enum Step { case microphone, download, downloading, finishing, failed }

    @Published var step: Step
    @Published var micStatus: AVAuthorizationStatus
    @Published var progress: DownloadProgress? = nil
    @Published var note = ""
    @Published var paused = false
    @Published var failure = ""
    @Published var freeBytes: Int64

    let files: [ModelFile]
    let totalBytes: Int64
    /// Descarga + margen para la conversión del traductor.
    var requiredFree: Int64 { totalBytes + 1_500_000_000 }
    var enoughSpace: Bool { freeBytes >= requiredFree }

    var onReady: () -> Void = {}
    var onQuit: () -> Void = {}
    private var downloader: ModelDownloader?

    init() {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        let list = ModelManifest.filesToDownload()
        micStatus = status
        step = status == .authorized ? .download : .microphone
        files = list
        totalBytes = list.reduce(0) { $0 + $1.size }
        freeBytes = ModelSetup.freeBytes()
    }

    // Micrófono

    func requestMicrophone() {
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
            DispatchQueue.main.async {
                guard let self else { return }
                self.micStatus = AVCaptureDevice.authorizationStatus(for: .audio)
                if granted { self.step = .download }
            }
        }
    }

    func openPrivacySettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
            NSWorkspace.shared.open(url)
        }
    }

    /// El permiso se puede dar después desde Ajustes del Sistema; no bloquea la descarga.
    func continueWithoutMicrophone() { step = .download }

    /// Al volver de Ajustes del Sistema, se vuelve a leer el estado.
    func refreshMicrophone() {
        micStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        if micStatus == .authorized && step == .microphone { step = .download }
    }

    // Descarga

    func startDownload() {
        freeBytes = ModelSetup.freeBytes()
        guard enoughSpace else { step = .download; return }
        step = .downloading
        paused = false
        note = ""
        progress = nil
        downloader?.shutdown()
        let d = ModelDownloader(files: files, hub: ModelManifest.hubDir()) { [weak self] event in self?.handle(event) }
        downloader = d
        d.start()
    }

    func togglePause() {
        if paused {
            paused = false
            note = ""
            downloader?.start()
        } else {
            downloader?.pause()
        }
    }

    private func handle(_ event: DownloadEvent) {
        switch event {
        case .progress(let p):
            progress = p
            if !paused { note = "" }
        case .verifying(let label):
            note = "Verificando «\(label)»…"
        case .retrying(let message):
            note = message
        case .paused:
            paused = true
            note = "En pausa. Lo ya descargado se conserva."
        case .failed(let message):
            fail(message)
        case .finished:
            finish()
        }
    }

    private func finish() {
        step = .finishing
        downloader?.shutdown()
        downloader = nil
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let problem = ModelSetup.prepareTranslator()
            DispatchQueue.main.async {
                if let problem { self?.fail(problem); return }
                ModelSetup.modelsReady { ok in
                    guard let self else { return }
                    if ok { self.onReady() } else {
                        self.fail("Los modelos se bajaron, pero la comprobación final no pasó. Reintentá la descarga; si se repite, escribinos.")
                    }
                }
            }
        }
    }

    private func fail(_ message: String) {
        downloader?.shutdown()
        downloader = nil
        failure = message
        step = .failed
    }
}

// MARK: - Pantalla (SwiftUI)

struct SetupView: View {
    @ObservedObject var model: SetupModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            content
            Spacer(minLength: 0)
            footer
        }
        .padding(28)
        .frame(width: 540, height: 400)
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            model.refreshMicrophone()
        }
    }

    @ViewBuilder private var content: some View {
        switch model.step {
        case .microphone: microphone
        case .download: download
        case .downloading: downloading
        case .finishing: finishing
        case .failed: failed
        }
    }

    // Paso 1: micrófono

    private var microphone: some View {
        VStack(alignment: .leading, spacing: 12) {
            Image(systemName: "mic.fill").font(.system(size: 34)).foregroundStyle(Color.accentColor)
            Text("Permiso del micrófono").font(.title2.weight(.semibold))
            Text("opentraductor necesita tu micrófono para escuchar lo que decís y traducirlo. El audio se procesa en tu Mac: no se graba ni se envía a ningún servidor. macOS te va a pedir confirmación.")
                .fixedSize(horizontal: false, vertical: true)
            if model.micStatus == .denied || model.micStatus == .restricted {
                Text("El permiso está desactivado. Podés activarlo en Ajustes del Sistema, en Privacidad y seguridad, Micrófono. También podés seguir y activarlo después.")
                    .font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // Paso 2: descarga

    private var download: some View {
        VStack(alignment: .leading, spacing: 12) {
            Image(systemName: "arrow.down.circle.fill").font(.system(size: 34)).foregroundStyle(Color.accentColor)
            Text("Descarga de modelos").font(.title2.weight(.semibold))
            Text("Para traducir sin internet, la app baja una sola vez los modelos de voz y de traducción (unos \(ModelSetup.format(model.totalBytes))) desde Hugging Face. Después funciona sin conexión. No se envía audio ni ningún dato tuyo. Conviene estar con wifi.")
                .fixedSize(horizontal: false, vertical: true)
            if model.freeBytes != Int64.max {
                Text(model.enoughSpace
                     ? "Espacio libre en este Mac: \(ModelSetup.format(model.freeBytes))."
                     : "Hace falta al menos \(ModelSetup.format(model.requiredFree)) libres y hay \(ModelSetup.format(model.freeBytes)). Liberá espacio y reintentá.")
                    .font(.callout)
                    .foregroundStyle(model.enoughSpace ? Color.secondary : Color.red)
            }
        }
    }

    private var downloading: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(model.paused ? "Descarga en pausa" : "Descargando modelos…").font(.title2.weight(.semibold))
            ProgressView(value: model.progress?.fraction ?? 0)
                .progressViewStyle(.linear)
            if let p = model.progress {
                Text(progressLine(p)).font(.callout).monospacedDigit()
                Text("Archivo \(p.fileIndex) de \(p.fileCount): \(p.fileLabel)").font(.callout).foregroundStyle(.secondary)
            } else {
                Text("Preparando…").font(.callout).foregroundStyle(.secondary)
            }
            if !model.note.isEmpty {
                Text(model.note).font(.callout).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Text("Podés pausar o cerrar la app: al volver, la descarga sigue desde donde quedó.")
                .font(.footnote).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private func progressLine(_ p: DownloadProgress) -> String {
        var s = "\(ModelSetup.format(p.doneBytes)) de \(ModelSetup.format(p.totalBytes))"
        if !model.paused && p.bytesPerSecond > 0 {
            s += " · \(ModelSetup.format(Int64(p.bytesPerSecond)))/s"
            if let r = p.remainingSeconds { s += " · \(Self.remaining(r))" }
        }
        return s
    }

    private static func remaining(_ seconds: Double) -> String {
        let minutes = Int((seconds / 60).rounded(.up))
        if minutes <= 1 { return "falta menos de 1 minuto" }
        if minutes > 120 { return "faltan más de 2 horas" }
        return "faltan unos \(minutes) minutos"
    }

    private var finishing: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Preparando el traductor…").font(.title2.weight(.semibold))
            ProgressView().progressViewStyle(.linear)
            Text("Se está verificando todo y dejando el traductor listo para tu Mac. Puede tardar un momento.")
                .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }

    private var failed: some View {
        VStack(alignment: .leading, spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 34)).foregroundStyle(.orange)
            Text("No se pudo completar la descarga").font(.title2.weight(.semibold))
            Text(model.failure).fixedSize(horizontal: false, vertical: true)
            Text("Lo ya descargado se conserva y se retoma al reintentar.")
                .font(.callout).foregroundStyle(.secondary)
        }
    }

    // Botones

    @ViewBuilder private var footer: some View {
        HStack {
            Button("Salir") { model.onQuit() }
            Spacer()
            switch model.step {
            case .microphone:
                if model.micStatus == .denied || model.micStatus == .restricted {
                    Button("Seguir sin micrófono") { model.continueWithoutMicrophone() }
                    Button("Abrir Ajustes del Sistema") { model.openPrivacySettings() }
                        .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                } else {
                    Button("Permitir micrófono") { model.requestMicrophone() }
                        .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                }
            case .download:
                Button("Descargar modelos") { model.startDownload() }
                    .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                    .disabled(!model.enoughSpace)
            case .downloading:
                Button(model.paused ? "Reanudar" : "Pausar") { model.togglePause() }
                    .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
            case .finishing:
                EmptyView()
            case .failed:
                Button("Reintentar") { model.startDownload() }
                    .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
            }
        }
        .controlSize(.large)
    }
}
