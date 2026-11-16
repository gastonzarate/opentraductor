import AppKit
import SwiftUI

// T-42: pantalla principal nativa, Ajustes y Acerca de.

/// Enlaces públicos que la app muestra (guía 5.1.1(i): la política de privacidad tiene que estar dentro de la app).
enum Links {
    static let privacy = URL(string: "https://opentraductor.lat/privacidad.html")!
    static let support = URL(string: "https://opentraductor.lat/soporte.html")!
}

struct MainView: View {
    @ObservedObject var model: AppModel
    @State private var showGuide = false

    var body: some View {
        switch model.phase {
        case .connecting(let text):
            VStack(spacing: 14) {
                ProgressView()
                Text(text).foregroundStyle(.secondary).multilineTextAlignment(.center)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(40)
        case .failed(let text):
            VStack(spacing: 12) {
                Image(systemName: "exclamationmark.triangle.fill").font(.system(size: 36)).foregroundStyle(.orange)
                Text("No se pudo iniciar").font(.title2.weight(.semibold))
                Text(text).textSelection(.enabled).multilineTextAlignment(.center)
                Text("Cerrá y volvé a abrir la app. Si se repite, escribinos con este detalle.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(40)
        case .ready:
            ready
        }
    }

    private var ready: some View {
        VStack(spacing: 0) {
            header
            Form {
                Section {
                    LabeledContent("Idioma") { Text("Español (Argentina) → Inglés (EE. UU.)").foregroundStyle(.secondary) }
                    Picker("Micrófono", selection: Binding(get: { model.inputID }, set: { model.setInput($0) })) {
                        ForEach(model.inputs) { Text($0.name).tag($0.id) }
                    }
                    Picker("Salida de la voz", selection: Binding(get: { model.outputID }, set: { model.setOutput($0) })) {
                        Text("Solo subtítulos (sin voz)").tag(AppModel.noOutputID)
                        ForEach(model.outputs) { Text($0.name).tag($0.id) }
                    }
                    if !model.onlySubtitles {
                        Text("Si la voz sale por parlantes, usá auriculares para que el micrófono no la vuelva a captar.")
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                    Button("Actualizar la lista de dispositivos") { model.refreshDevices() }
                        .buttonStyle(.link)
                }
                .disabled(model.isActive)
            }
            .formStyle(.grouped)
            .scrollDisabled(true)
            .frame(height: model.onlySubtitles ? 220 : 255)
            if let banner = bannerText { banner_(banner) }
            Divider()
            subtitles
            Divider()
            footer
        }
        .frame(minWidth: 720, minHeight: 540)
        .sheet(isPresented: $showGuide) { BlackHoleGuide(model: model) }
    }

    // MARK: Encabezado

    private var header: some View {
        HStack(spacing: 12) {
            Circle().fill(stateColor).frame(width: 10, height: 10).accessibilityHidden(true)
            Text(model.stateText).font(.headline)
            Spacer()
            Button {
                model.toggle()
            } label: {
                Label(model.state == "running" || model.state == "starting" ? "Detener" : "Iniciar",
                      systemImage: model.state == "running" || model.state == "starting" ? "stop.fill" : "play.fill")
                    .frame(minWidth: 90)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .keyboardShortcut(.defaultAction)
            .disabled(model.busy || model.state == "stopping" || model.inputID.isEmpty || !model.missingChecks.isEmpty)
        }
        .padding(.horizontal, 20)
        .padding(.top, 16)
        .padding(.bottom, 4)
    }

    private var stateColor: Color {
        switch model.state {
        case "running": return .green
        case "starting", "stopping": return .orange
        case "error": return .red
        default: return .secondary
        }
    }

    private var bannerText: String? {
        if !model.missingChecks.isEmpty { return "Faltan modelos o componentes: \(model.missingChecks.joined(separator: ", ")). Reiniciá la app para volver a la descarga guiada." }
        if let e = model.engineError, model.state == "error" { return e }
        return model.notice
    }

    private func banner_(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
            Text(text).font(.callout).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
            Spacer()
            if model.notice != nil {
                Button("Cerrar") { model.notice = nil }.buttonStyle(.link)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
    }

    // MARK: Subtítulos

    private var subtitles: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    if model.lines.isEmpty && model.partialES.isEmpty {
                        Text(model.state == "running"
                             ? "Hablá en español. La traducción aparece acá."
                             : "Elegí tu micrófono y tocá «Iniciar». Los subtítulos aparecen acá.")
                            .foregroundStyle(.secondary)
                    }
                    ForEach(model.lines) { line in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(line.es).foregroundStyle(.secondary)
                            Text(line.en).font(.title3.weight(.medium))
                        }
                        .textSelection(.enabled)
                        .accessibilityElement(children: .combine)
                    }
                    if !model.partialES.isEmpty {
                        Text(model.partialES).italic().foregroundStyle(.secondary)
                    }
                    Color.clear.frame(height: 1).id("fin")
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .onChange(of: model.lines.count) { _, _ in proxy.scrollTo("fin") }
            .onChange(of: model.partialES) { _, _ in proxy.scrollTo("fin") }
        }
        .frame(maxHeight: .infinity)
    }

    // MARK: Pie

    private var footer: some View {
        HStack(spacing: 14) {
            Label(latencyText, systemImage: "timer")
                .font(.callout).foregroundStyle(.secondary)
                .help("Tiempo medido entre el fin de tu frase y el primer audio de la voz traducida. No se mide con solo subtítulos.")
            Spacer()
            Button("Usar con Meet, Zoom o Discord…") {
                model.loadVirtualMic()
                showGuide = true
            }
            .buttonStyle(.link)
            Text("Todo se procesa en tu Mac.").font(.callout).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
    }

    private var latencyText: String {
        if model.onlySubtitles { return "Latencia: sin voz, solo subtítulos" }
        if let ms = model.latencyMs { return "Latencia medida: \(Int(ms)) ms" }
        return "Latencia: se mide al traducir"
    }
}

// MARK: - Guía opcional de BlackHole

struct BlackHoleGuide: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Usar opentraductor en Meet, Zoom o Discord").font(.title2.weight(.semibold))
            Text("Esto es opcional. Sin BlackHole la app funciona igual: la voz sale por tus parlantes o auriculares y ves los subtítulos.")
                .fixedSize(horizontal: false, vertical: true)
            if let v = model.virtualMic {
                if v.installed {
                    Label("BlackHole 2ch está instalado.", systemImage: "checkmark.circle.fill").foregroundStyle(.green)
                    Button("Usar BlackHole 2ch como salida") { model.useBlackHole(); dismiss() }
                        .buttonStyle(.borderedProminent)
                        .disabled(model.isActive)
                    Text("Después, en la reunión, elegí «BlackHole 2ch» como micrófono.").foregroundStyle(.secondary)
                } else {
                    Text("BlackHole 2ch no está instalado. Es un driver gratuito de terceros que instalás vos: opentraductor no instala nada.")
                        .fixedSize(horizontal: false, vertical: true)
                    ForEach(v.steps) { step in
                        VStack(alignment: .leading, spacing: 4) {
                            Text("\(step.n). \(step.title)").font(.headline)
                            Text(step.detail).fixedSize(horizontal: false, vertical: true)
                            if let cmd = step.command {
                                HStack {
                                    Text(cmd).font(.system(.body, design: .monospaced)).textSelection(.enabled)
                                    Button("Copiar") {
                                        NSPasteboard.general.clearContents()
                                        NSPasteboard.general.setString(cmd, forType: .string)
                                    }
                                }
                            }
                            if let u = step.url, let url = URL(string: u) { Link("Abrir la página de descarga", destination: url) }
                        }
                    }
                }
            } else {
                ProgressView()
            }
            HStack {
                Spacer()
                Button("Cerrar") { dismiss() }.keyboardShortcut(.cancelAction)
            }
        }
        .padding(24)
        .frame(width: 540)
    }
}

// MARK: - Ajustes

struct SettingsView: View {
    @ObservedObject var model: AppModel
    @State private var confirmDelete = false

    var body: some View {
        Form {
            Section("Voz") {
                Picker("Voz en inglés", selection: Binding(get: { model.voiceID }, set: { model.setVoice($0) })) {
                    ForEach(model.voices) { Text($0.label).tag($0.id) }
                }
                .disabled(model.isActive)
            }
            Section("Traducción") {
                VStack(alignment: .leading) {
                    Text("Pausa para cerrar una frase")
                    Slider(value: $model.endpointMs, in: 60...2000, step: 20, onEditingChanged: { editing in
                        if !editing { model.commitEndpoint() }
                    })
                    Text("\(Int(model.endpointMs)) ms de silencio. Menos pausa corta las frases antes; más pausa espera más.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                Picker("Búfer de audio", selection: Binding(get: { model.audioLatency }, set: { model.setAudioLatency($0) })) {
                    Text("Normal").tag("default")
                    Text("Bajo").tag("low")
                    Text("Alto").tag("high")
                }
                .disabled(model.isActive)
            }
            Section("Privacidad") {
                Text("El audio se procesa en tu Mac y no se graba. Se guardan solo los textos de cada sesión, en tu Mac.")
                    .font(.callout).foregroundStyle(.secondary)
                Link("Política de privacidad", destination: Links.privacy)
                Button("Abrir permisos del micrófono…") { model.openMicrophoneSettings() }
                Button("Borrar historial de textos…", role: .destructive) { confirmDelete = true }
                    .disabled(model.isActive)
            }
        }
        .formStyle(.grouped)
        .frame(width: 480, height: 490)
        .confirmationDialog("¿Borrar todo el historial de textos?", isPresented: $confirmDelete) {
            Button("Borrar", role: .destructive) { model.deleteHistory() }
            Button("Cancelar", role: .cancel) {}
        } message: {
            Text("Se borran los textos de todas las sesiones. No se puede deshacer.")
        }
    }
}

// MARK: - Acerca de

struct AboutView: View {
    private var version: String {
        let v = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? ""
        let b = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? ""
        return b.isEmpty ? "Versión \(v)" : "Versión \(v) (\(b))"
    }

    private let licenses: [(String, String)] = [
        ("Parakeet TDT 0.6B v3 (NVIDIA): reconocimiento de voz", "Creative Commons Atribución 4.0 (CC-BY 4.0)"),
        ("Kokoro-82M: voz en inglés", "Apache 2.0"),
        ("Helsinki-NLP opus-mt-es-en (Marian): traducción", "Creative Commons Atribución 4.0 (CC-BY 4.0)"),
        ("MLX (Apple)", "MIT"),
        ("CTranslate2", "MIT"),
        ("Python", "Python Software Foundation License"),
        ("Moonshine (experimental)", "Moonshine AI Community License"),
        ("BlackHole (opcional, no incluido)", "GPL-3.0; lo instala la persona por su cuenta"),
    ]

    var body: some View {
        VStack(spacing: 12) {
            Image(nsImage: NSApp.applicationIconImage).resizable().frame(width: 72, height: 72)
            Text("opentraductor").font(.title.weight(.semibold))
            Text(version).foregroundStyle(.secondary)
            Text("Traductor de voz de español (Argentina) a inglés (EE. UU.) que corre en tu Mac. Sin telemetría. No se graba audio. La única conexión a internet es la descarga inicial de modelos.")
                .multilineTextAlignment(.center).fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 16) {
                Link("Política de privacidad", destination: Links.privacy)
                Link("Soporte", destination: Links.support)
            }
            Divider()
            Text("Licencias de terceros").font(.headline).frame(maxWidth: .infinity, alignment: .leading)
            ScrollView {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(licenses, id: \.0) { item in
                        VStack(alignment: .leading, spacing: 1) {
                            Text(item.0)
                            Text(item.1).font(.callout).foregroundStyle(.secondary)
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 170)
        }
        .padding(24)
        .frame(width: 460)
    }
}
