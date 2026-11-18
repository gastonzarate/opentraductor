import AppKit
import SwiftUI

/// Shell nativo de opentraductor: lanza el runtime Python embebido (solo en 127.0.0.1) y muestra una interfaz SwiftUI propia.
/// La única conexión a internet es la descarga inicial de modelos (ModelSetup.swift, con pantalla explícita).
/// El motor corre siempre en modo offline.

final class RuntimeProcess {
    private let process = Process()
    private let outPipe = Pipe()
    private let errPipe = Pipe()
    private(set) var stderrTail = ""
    var onURL: ((URL) -> Void)?
    var onExit: ((Int32) -> Void)?

    /// Build de la Mac App Store (Info.plist: OTEdition = store): sin licencia propia, sin activación.
    static var isStoreEdition: Bool { (Bundle.main.object(forInfoDictionaryKey: "OTEdition") as? String) == "store" }

    static func supportDir() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return base.appendingPathComponent("OpenTraductor", isDirectory: true)
    }

    static func pythonURL() -> URL { Bundle.main.resourceURL!.appendingPathComponent("python/bin/python3.12") }
    static func appDirURL() -> URL { Bundle.main.resourceURL!.appendingPathComponent("app") }

    /// Entorno común del motor Python. Los datos viven en el contenedor del sandbox (Application Support).
    static func makeEnvironment() throws -> [String: String] {
        let appDir = appDirURL()
        let support = supportDir()
        let fm = FileManager.default
        for sub in ["data", "models", "hf", "moonshine"] {
            try fm.createDirectory(at: support.appendingPathComponent(sub), withIntermediateDirectories: true)
        }
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"  // el bundle es de solo lectura y está firmado
        env["PYTHONPATH"] = appDir.path
        env["OPENTRADUCTOR_DATA"] = support.appendingPathComponent("data").path
        env["OPENTRADUCTOR_MODELS"] = support.appendingPathComponent("models").path
        env["OPENTRADUCTOR_MOONSHINE_DIR"] = support.appendingPathComponent("moonshine").path
        env["HF_HOME"] = support.appendingPathComponent("hf").path
        // Marian ya convertido a CTranslate2 viene dentro del bundle (solo lectura); el build lo genera sin torch en la app.
        let marian = Bundle.main.resourceURL!.appendingPathComponent("marian_ct2")
        if fm.fileExists(atPath: marian.appendingPathComponent("model.bin").path) {
            env["OPENTRADUCTOR_MARIAN_CT2"] = marian.path
        }
        if isStoreEdition { env["OPENTRADUCTOR_EDITION"] = "store" }
        // T-49: motores del build (Info.plist). Qwen lee sus pesos del snapshot que baja la descarga guiada.
        let profile = EngineProfile.current
        env["OPENTRADUCTOR_ASR_BACKEND"] = profile.asrBackend
        env["OPENTRADUCTOR_MT_ENGINE"] = profile.mtEngine
        if profile.mtEngine == "qwen", let q = ModelManifest.qwenDir(profile.qwenModel) { env["OPENTRADUCTOR_QWEN_DIR"] = q.path }
        // BUG-7 (22e7f7a, validado por @qa en texto): normaliza cifras y traduce oración por oración.
        env["OPENTRADUCTOR_MT_NORMALIZE"] = "1"
        return env
    }

    func start() throws {
        let python = Self.pythonURL()
        let appDir = Self.appDirURL()
        var env = try Self.makeEnvironment()
        env["HF_HUB_OFFLINE"] = "1"  // el motor nunca descarga
        env["TRANSFORMERS_OFFLINE"] = "1"
        process.executableURL = python
        process.arguments = EngineProfile.current.runtimeArgs("run") + ["--port", "0"]
        process.currentDirectoryURL = appDir
        process.environment = env
        process.standardOutput = outPipe
        process.standardError = errPipe
        process.terminationHandler = { [weak self] p in
            DispatchQueue.main.async { self?.onExit?(p.terminationStatus) }
        }
        outPipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            for line in text.split(separator: "\n") {
                if let r = line.range(of: "http://127.0.0.1:"), let url = URL(string: String(line[r.lowerBound...]).split(separator: " ").first.map(String.init) ?? "") {
                    DispatchQueue.main.async { self?.onURL?(url) }
                }
            }
        }
        errPipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let data = h.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            DispatchQueue.main.async {
                guard let self else { return }
                self.stderrTail = String((self.stderrTail + text).suffix(2000))
            }
        }
        try process.run()
    }

    func stop() {
        guard process.isRunning else { return }
        process.terminate()
        let deadline = Date().addingTimeInterval(3)
        while process.isRunning && Date() < deadline { usleep(50_000) }
        if process.isRunning { kill(process.processIdentifier, SIGKILL) }
    }
}

// T-42: ventana principal, Ajustes y Acerca de en SwiftUI (sin WebKit). Ver MainView.swift, AppModel.swift y EngineClient.swift.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuItemValidation {
    private var window: NSWindow!
    private let model = AppModel()
    private let runtime = RuntimeProcess()
    private var started = false
    private var quitting = false
    private var modelSetup: ModelSetup?
    private var settingsWindow: NSWindow?
    private var aboutWindow: NSWindow?
    private var toggleItem: NSMenuItem?

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildMenu()
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 760, height: 640),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "opentraductor"
        window.minSize = NSSize(width: 720, height: 540)
        window.contentView = NSHostingView(rootView: MainView(model: model))
        window.setFrameAutosaveName("main")
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        runtime.onURL = { [weak self] url in
            guard let self, !self.started else { return }
            self.started = true
            self.model.attach(url: url)
        }
        runtime.onExit = { [weak self] code in
            guard let self, !self.quitting else { return }
            self.model.showFailure("El motor se detuvo (código \(code)).\n\n" + self.runtime.stderrTail)
        }
        // Primer arranque: si faltan los modelos, descarga guiada antes de iniciar el motor.
        model.showMessage("Comprobando los modelos de voz…")
        ModelSetup.modelsReady { [weak self] ready in
            guard let self else { return }
            if ready { return ModelSetup.requestMicrophoneIfNeeded { self.startRuntime() } }
            self.model.showMessage("Faltan los modelos de voz. Hace falta una descarga inicial.")
            let setup = ModelSetup(parent: self.window) { [weak self] in self?.startRuntime() }
            self.modelSetup = setup
            setup.present()
        }
    }

    private func startRuntime() {
        modelSetup = nil
        model.showMessage("Preparando el motor de traducción en tu Mac…")
        do { try runtime.start() } catch {
            model.showFailure("No se pudo iniciar el motor: \(error.localizedDescription)")
        }
    }

    private func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem(); main.addItem(appItem)
        let appMenu = NSMenu(); appItem.submenu = appMenu
        appMenu.addItem(withTitle: "Acerca de opentraductor", action: #selector(showAbout), keyEquivalent: "").target = self
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Ajustes…", action: #selector(showSettings), keyEquivalent: ",").target = self
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Ocultar opentraductor", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Salir de opentraductor", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let editItem = NSMenuItem(); main.addItem(editItem)
        let edit = NSMenu(title: "Edición"); editItem.submenu = edit
        edit.addItem(withTitle: "Cortar", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copiar", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Pegar", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Seleccionar todo", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        let trItem = NSMenuItem(); main.addItem(trItem)
        let tr = NSMenu(title: "Traducción"); trItem.submenu = tr
        toggleItem = tr.addItem(withTitle: "Iniciar", action: #selector(toggleTranslation), keyEquivalent: "r")
        toggleItem?.target = self
        let refresh = tr.addItem(withTitle: "Actualizar la lista de dispositivos", action: #selector(refreshDevices), keyEquivalent: "r")
        refresh.keyEquivalentModifierMask = [.command, .shift]
        refresh.target = self
        let winItem = NSMenuItem(); main.addItem(winItem)
        let win = NSMenu(title: "Ventana"); winItem.submenu = win
        win.addItem(withTitle: "Minimizar", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        win.addItem(withTitle: "Cerrar", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        let helpItem = NSMenuItem(); main.addItem(helpItem)
        let help = NSMenu(title: "Ayuda"); helpItem.submenu = help
        help.addItem(withTitle: "Política de privacidad", action: #selector(openPrivacy), keyEquivalent: "").target = self
        help.addItem(withTitle: "Soporte", action: #selector(openSupport), keyEquivalent: "").target = self
        help.addItem(withTitle: "Licencias de terceros", action: #selector(showAbout), keyEquivalent: "").target = self
        NSApp.helpMenu = help
        NSApp.mainMenu = main
    }

    @objc private func toggleTranslation() { model.toggle() }
    @objc private func refreshDevices() { model.refreshDevices() }
    @objc private func openPrivacy() { NSWorkspace.shared.open(Links.privacy) }
    @objc private func openSupport() { NSWorkspace.shared.open(Links.support) }

    func validateMenuItem(_ item: NSMenuItem) -> Bool {
        switch item.action {
        case #selector(toggleTranslation):
            let running = model.state == "running" || model.state == "starting"
            item.title = running ? "Detener" : "Iniciar"
            guard case .ready = model.phase else { return false }
            return !model.busy && model.state != "stopping" && (running || (!model.inputID.isEmpty && model.missingChecks.isEmpty))
        case #selector(refreshDevices):
            guard case .ready = model.phase else { return false }
            return !model.isActive
        default:
            return true
        }
    }

    @objc private func showSettings() {
        if settingsWindow == nil {
            let w = NSWindow(contentViewController: NSHostingController(rootView: SettingsView(model: model)))
            w.title = "Ajustes"
            w.styleMask = [.titled, .closable]
            w.isReleasedWhenClosed = false
            settingsWindow = w
        }
        settingsWindow?.center()
        settingsWindow?.makeKeyAndOrderFront(nil)
    }

    @objc private func showAbout() {
        if aboutWindow == nil {
            let w = NSWindow(contentViewController: NSHostingController(rootView: AboutView()))
            w.title = "Acerca de opentraductor"
            w.styleMask = [.titled, .closable]
            w.isReleasedWhenClosed = false
            aboutWindow = w
        }
        aboutWindow?.center()
        aboutWindow?.makeKeyAndOrderFront(nil)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        quitting = true
        runtime.stop()
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = MainActor.assumeIsolated { AppDelegate() }
app.delegate = delegate
signal(SIGTERM, SIG_IGN)
let termSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
termSource.setEventHandler { NSApp.terminate(nil) }  // cierra también el runtime hijo
termSource.resume()
app.run()
