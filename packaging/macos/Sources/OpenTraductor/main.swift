import AppKit
import WebKit

/// Shell nativo de opentraductor: lanza el runtime Python embebido (solo en 127.0.0.1) y muestra su UI en una ventana propia.
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
        if isStoreEdition { env["OPENTRADUCTOR_EDITION"] = "store" }
        return env
    }

    func start() throws {
        let python = Self.pythonURL()
        let appDir = Self.appDirURL()
        var env = try Self.makeEnvironment()
        env["HF_HUB_OFFLINE"] = "1"  // el motor nunca descarga
        env["TRANSFORMERS_OFFLINE"] = "1"
        process.executableURL = python
        process.arguments = ["-m", "runtime", "run", "--port", "0"]
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

final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate {
    private var window: NSWindow!
    private var web: WKWebView!
    private let runtime = RuntimeProcess()
    private var baseURL: URL?
    private var quitting = false
    private var modelSetup: ModelSetup?

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildMenu()
        let cfg = WKWebViewConfiguration()
        cfg.websiteDataStore = .nonPersistent()  // sin cookies ni caché en disco
        web = WKWebView(frame: .zero, configuration: cfg)
        web.navigationDelegate = self
        web.uiDelegate = self
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1100, height: 780),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "opentraductor"
        window.minSize = NSSize(width: 760, height: 560)
        window.contentView = web
        window.setFrameAutosaveName("main")
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        showMessage(title: "Iniciando…", body: "Preparando el motor de traducción en tu Mac.")

        runtime.onURL = { [weak self] url in
            guard let self, self.baseURL == nil else { return }
            self.baseURL = url
            self.web.load(URLRequest(url: url))
        }
        runtime.onExit = { [weak self] code in
            guard let self, !self.quitting else { return }
            self.showMessage(title: "El motor se detuvo (código \(code))",
                             body: "Cerrá y volvé a abrir la app. Si se repite, escribinos con este detalle:\n\n" + self.runtime.stderrTail)
        }
        // Primer arranque: si faltan los modelos, descarga guiada antes de iniciar el motor.
        ModelSetup.modelsReady { [weak self] ready in
            guard let self else { return }
            if ready { return ModelSetup.requestMicrophoneIfNeeded { self.startRuntime() } }
            self.showMessage(title: "Faltan los modelos de voz", body: "Hace falta una descarga inicial.")
            let setup = ModelSetup(parent: self.window) { [weak self] in self?.startRuntime() }
            self.modelSetup = setup
            setup.present()
        }
    }

    private func startRuntime() {
        modelSetup = nil
        showMessage(title: "Iniciando…", body: "Preparando el motor de traducción en tu Mac.")
        do { try runtime.start() } catch {
            showMessage(title: "No se pudo iniciar el motor", body: "\(error.localizedDescription)")
        }
    }

    private func esc(_ s: String) -> String {
        s.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;")
    }

    private func showMessage(title: String, body: String) {
        let html = """
        <html><body style="font:15px -apple-system;padding:48px;color-scheme:light dark">
        <h2>\(esc(title))</h2><pre style="white-space:pre-wrap;font:13px ui-monospace">\(esc(body))</pre></body></html>
        """
        web.loadHTMLString(html, baseURL: nil)
    }

    // Solo se navega dentro del servidor local; todo lo demás (p. ej. existential.audio) se abre en el navegador.
    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = action.request.url else { return decisionHandler(.cancel) }
        if url.scheme == "about" || url.host == "127.0.0.1" { return decisionHandler(.allow) }
        if ["http", "https"].contains(url.scheme ?? "") { NSWorkspace.shared.open(url) }
        decisionHandler(.cancel)
    }

    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration, for action: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let url = action.request.url, ["http", "https"].contains(url.scheme ?? "") { NSWorkspace.shared.open(url) }
        return nil
    }

    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
        let a = NSAlert(); a.messageText = message; a.runModal(); completionHandler()
    }

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        let a = NSAlert(); a.messageText = message
        a.addButton(withTitle: "Aceptar"); a.addButton(withTitle: "Cancelar")
        completionHandler(a.runModal() == .alertFirstButtonReturn)
    }

    private func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem(); main.addItem(appItem)
        let appMenu = NSMenu(); appItem.submenu = appMenu
        appMenu.addItem(withTitle: "Acerca de opentraductor", action: #selector(about), keyEquivalent: "").target = self
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Ocultar opentraductor", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Salir de opentraductor", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        // Menú Edición: sin él, Cmd+C/V no funcionan en el campo de la clave de licencia.
        let editItem = NSMenuItem(); main.addItem(editItem)
        let edit = NSMenu(title: "Edición"); editItem.submenu = edit
        edit.addItem(withTitle: "Deshacer", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cortar", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copiar", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Pegar", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Seleccionar todo", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        let viewItem = NSMenuItem(); main.addItem(viewItem)
        let view = NSMenu(title: "Ventana"); viewItem.submenu = view
        view.addItem(withTitle: "Minimizar", action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        view.addItem(withTitle: "Cerrar", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        NSApp.mainMenu = main
    }

    @objc private func about() {
        let net = RuntimeProcess.isStoreEdition
            ? "La única conexión a internet es la descarga inicial de modelos."
            : "La única conexión a internet es la activación de la licencia, una vez, y la descarga inicial de modelos."
        let credits = NSAttributedString(string: """
        Traductor de voz español (Argentina) a inglés (EE. UU.), 100 % en tu Mac. Sin telemetría; no se graba audio.
        \(net)

        Modelos de voz Moonshine: Moonshine AI Community License. Parakeet (NVIDIA): CC-BY 4.0.
        """)
        NSApp.orderFrontStandardAboutPanel(options: [.credits: credits])
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        quitting = true
        runtime.stop()
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = AppDelegate()
app.delegate = delegate
signal(SIGTERM, SIG_IGN)
let termSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
termSource.setEventHandler { NSApp.terminate(nil) }  // cierra también el runtime hijo
termSource.resume()
app.run()
