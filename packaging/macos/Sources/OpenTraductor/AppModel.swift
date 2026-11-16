import AppKit
import SwiftUI

/// Estado de la pantalla principal. Habla con el motor por EngineClient; todo corre en el hilo principal.
@MainActor
final class AppModel: ObservableObject {
    enum Phase {
        case connecting(String)
        case ready
        case failed(String)
    }

    struct Line: Identifiable {
        let id = UUID()
        let es: String
        let en: String
    }

    /// Valor de `output_id` que el motor entiende como «solo subtítulos» (runtime/service.py, NO_OUTPUT).
    static let noOutputID = "__none__"

    @Published var phase: Phase = .connecting("Preparando el motor de traducción en tu Mac…")
    @Published var state = "idle"
    @Published var engineError: String?
    @Published var notice: String?
    @Published var inputs: [AudioDevice] = []
    @Published var outputs: [AudioDevice] = []
    @Published var voices: [VoiceOption] = []
    @Published var inputID = ""
    @Published var outputID = ""
    @Published var voiceID = ""
    @Published var endpointMs: Double = 500
    @Published var audioLatency = "default"
    @Published var lines: [Line] = []
    @Published var partialES = ""
    @Published var latencyMs: Double?
    @Published var missingChecks: [String] = []
    @Published var virtualMic: VirtualMicInfo?
    @Published var busy = false

    private var client: EngineClient?
    private var confirmedOutputID: String?
    private var eventsTask: Task<Void, Never>?

    var isActive: Bool { state == "starting" || state == "running" || state == "stopping" }
    var onlySubtitles: Bool { outputID == Self.noOutputID }

    var stateText: String {
        switch state {
        case "starting": return "Cargando…"
        case "running": return "Traduciendo"
        case "stopping": return "Deteniendo…"
        case "error": return "Error"
        default: return "Detenido"
        }
    }

    // MARK: Conexión

    func showMessage(_ text: String) { phase = .connecting(text) }
    func showFailure(_ text: String) { phase = .failed(text) }

    /// Se llama cuando el motor ya imprimió su URL local.
    func attach(url: URL) {
        let c = EngineClient(base: url)
        client = c
        Task {
            do {
                try await c.connect()
                try await reload(c)
                phase = .ready
                listen(c)
            } catch {
                phase = .failed("No se pudo conectar con el motor de traducción: \(error.localizedDescription)")
            }
        }
    }

    private func reload(_ c: EngineClient) async throws {
        let st = try await c.status()
        let dv = try await c.devices()
        let vs = try await c.voices()
        inputs = dv.inputs
        outputs = dv.outputs
        voices = vs.voices
        apply(st)
        var patch: [String: Any] = [:]
        let cfg = st.config
        // Entrada: la guardada si sigue existiendo; si no, el micrófono por defecto (mejor uno que no sea virtual).
        // Pasos separados (no una cadena de `??`): el compilador tarda o falla con expresiones largas.
        var wantIn = ""
        if let id = inputs.first(where: { $0.id == cfg?.input_id })?.id {
            wantIn = id
        } else if let id = inputs.first(where: { $0.isDefault && !$0.isVirtual })?.id {
            wantIn = id
        } else if let id = inputs.first(where: { !$0.isVirtual })?.id {
            wantIn = id
        } else if let id = inputs.first?.id {
            wantIn = id
        }
        // Salida: la guardada si sigue existiendo; si no, «solo subtítulos». Nunca se elige sola una salida real:
        // con parlantes y micrófono del equipo, la voz traducida se vuelve a captar.
        var wantOut = Self.noOutputID
        if cfg?.output_id == Self.noOutputID {
            wantOut = Self.noOutputID
        } else if let id = outputs.first(where: { $0.id == cfg?.output_id })?.id {
            wantOut = id
        }
        var wantVoice = ""
        if let id = voices.first(where: { $0.id == cfg?.voice })?.id {
            wantVoice = id
        } else if let id = voices.first?.id {
            wantVoice = id
        }
        if wantIn != cfg?.input_id, !wantIn.isEmpty { patch["input_id"] = wantIn }
        if wantOut != cfg?.output_id { patch["output_id"] = wantOut }
        if wantVoice != cfg?.voice, !wantVoice.isEmpty { patch["voice"] = wantVoice }
        inputID = wantIn
        outputID = wantOut
        voiceID = wantVoice
        if !patch.isEmpty, let updated = try? await c.configure(patch) { apply(updated) }
    }

    private func apply(_ st: EngineStatus) {
        state = st.state
        engineError = st.error
        if let ms = st.metrics?.last_endpoint_to_audio_ms { latencyMs = ms }
        missingChecks = (st.preflight ?? []).filter { !$0.ok }.map { $0.name }
        if let v = st.config?.endpoint_silence_ms { endpointMs = v }
        if let m = st.config?.audio_latency { audioLatency = m }
    }

    private func listen(_ c: EngineClient) {
        eventsTask?.cancel()
        eventsTask = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    for try await ev in c.events() { self?.handle(ev) }
                } catch {}
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    private func handle(_ ev: EngineEvent) {
        switch ev {
        case .state(let s, let err):
            state = s
            engineError = err
            if s == "idle" || s == "error" { partialES = "" }
            if s == "running" { notice = nil }
        case .partial(let es):
            partialES = es
        case .segment(let es, let en):
            partialES = ""
            lines.append(Line(es: es, en: en))
            if lines.count > 200 { lines.removeFirst(lines.count - 200) }
        case .latency(let ms):
            if let ms { latencyMs = ms }
        case .problem(let m):
            notice = m
        case .overflow:
            notice = "Se descartó un poco de audio porque la traducción no alcanzó a procesarlo a tiempo."
        }
    }

    // MARK: Acciones

    func toggle() {
        guard let c = client, !busy else { return }
        let stopping = state == "running" || state == "starting"
        if !stopping && !confirmHeadphones() { return }
        busy = true
        Task {
            do {
                if stopping {
                    try await c.stop()
                } else {
                    notice = nil
                    // Como la UI web: se aplica la elección y recién después se inicia.
                    var cfg: [String: Any] = ["input_id": inputID, "output_id": outputID]
                    if !voiceID.isEmpty { cfg["voice"] = voiceID }
                    apply(try await c.configure(cfg))
                    try await c.start()
                }
            } catch {
                notice = error.localizedDescription
            }
            if let st = try? await c.status() { apply(st) }
            busy = false
        }
    }

    /// Hay camino acústico posible: micrófono real y salida real (ni virtual ni «solo subtítulos»).
    private var feedbackPossible: Bool {
        guard !onlySubtitles,
              let out = outputs.first(where: { $0.id == outputID }),
              let inp = inputs.first(where: { $0.id == inputID }) else { return false }
        return !out.isVirtual && !inp.isVirtual
    }

    /// Pregunta una vez por salida si son auriculares. Devuelve false si la persona cancela.
    private func confirmHeadphones() -> Bool {
        guard feedbackPossible, confirmedOutputID != outputID else { return true }
        let name = outputs.first(where: { $0.id == outputID })?.name ?? "esta salida"
        let alert = NSAlert()
        alert.messageText = "¿«\(name)» son auriculares?"
        alert.informativeText = "Si la voz traducida sale por parlantes, el micrófono la capta y el sistema la traduce otra vez, y puede inventar frases. Con auriculares no pasa. También podés seguir solo con subtítulos."
        alert.addButton(withTitle: "Son auriculares")
        alert.addButton(withTitle: "Solo subtítulos")
        alert.addButton(withTitle: "Cancelar")
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            confirmedOutputID = outputID
            return true
        case .alertSecondButtonReturn:
            outputID = Self.noOutputID
            return true
        default:
            return false
        }
    }

    func push(_ patch: [String: Any]) {
        guard let c = client else { return }
        Task {
            do { apply(try await c.configure(patch)) } catch { notice = error.localizedDescription }
        }
    }

    func setInput(_ id: String) { inputID = id; push(["input_id": id]) }
    func setOutput(_ id: String) { outputID = id; push(["output_id": id]) }
    func setVoice(_ id: String) { voiceID = id; push(["voice": id]) }
    func setAudioLatency(_ mode: String) { audioLatency = mode; push(["audio_latency": mode]) }
    func commitEndpoint() { push(["endpoint_silence_ms": endpointMs]) }

    func refreshDevices() {
        guard let c = client, !isActive else { return }
        Task {
            do {
                let dv = try await c.refreshDevices()
                inputs = dv.inputs
                outputs = dv.outputs
                if !inputs.contains(where: { $0.id == inputID }) { inputID = inputs.first { $0.isDefault }?.id ?? inputs.first?.id ?? "" }
                if outputID != Self.noOutputID, !outputs.contains(where: { $0.id == outputID }) {
                    outputID = outputs.first { $0.isDefault }?.id ?? Self.noOutputID
                }
                push(["input_id": inputID, "output_id": outputID])
            } catch {
                notice = error.localizedDescription
            }
        }
    }

    func loadVirtualMic() {
        guard let c = client else { return }
        virtualMic = nil
        Task { virtualMic = try? await c.virtualMic() }
    }

    func useBlackHole() {
        if let id = virtualMic?.output_id { setOutput(id) }
    }

    func deleteHistory() {
        guard let c = client else { return }
        Task {
            do { try await c.deleteHistory(); notice = "Se borró el historial de textos." } catch { notice = error.localizedDescription }
        }
    }

    func openMicrophoneSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone") {
            NSWorkspace.shared.open(url)
        }
    }
}
