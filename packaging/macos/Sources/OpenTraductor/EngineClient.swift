import Foundation

// T-42 (en preparación, sin compilar todavía): cliente de la API HTTP local del motor (runtime/http_control.py).
// El token de sesión se lee del <meta name="ot-token"> que el servidor inyecta en GET / (no hace falta token para esa ruta).

struct EngineError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

struct AudioDevice: Decodable, Identifiable, Hashable {
    let id: String
    let name: String
    let isDefault: Bool
    let isVirtual: Bool

    enum CodingKeys: String, CodingKey { case id, name, isDefault = "default", isVirtual = "virtual" }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        isDefault = (try? c.decode(Bool.self, forKey: .isDefault)) ?? false
        isVirtual = (try? c.decode(Bool.self, forKey: .isVirtual)) ?? false
    }
}

struct DeviceList: Decodable {
    var inputs: [AudioDevice]
    var outputs: [AudioDevice]
}

struct VoiceOption: Decodable, Identifiable, Hashable {
    let id: String
    let label: String
}

struct VoiceList: Decodable { var voices: [VoiceOption] }

struct VirtualMicInfo: Decodable {
    struct Step: Decodable, Identifiable {
        var n: Int
        var title: String
        var detail: String
        var command: String?
        var url: String?
        var id: Int { n }
    }
    var installed: Bool
    var output_id: String?
    var steps: [Step]
    var download_url: String?
}

struct EngineStatus: Decodable {
    struct Check: Decodable { var name: String; var ok: Bool; var detail: String? }
    struct Config: Decodable {
        var input_id: String?
        var output_id: String?
        var voice: String?
        var endpoint_silence_ms: Double?
        var audio_latency: String?
    }
    struct Metrics: Decodable {
        var last_endpoint_to_audio_ms: Double?
        var overflow_count: Int?
    }
    var state: String
    var error: String?
    var preflight: [Check]?
    var config: Config?
    var metrics: Metrics?
}

enum EngineEvent {
    case state(String, String?)
    /// `id` = seg_id (nil con un motor viejo); `from` = from_seg_id (fila «live» que se reasigna).
    case partial(id: String?, from: String?, es: String)
    case segment(id: String?, from: String?, es: String, en: String)
    case discarded(id: String?, from: String?, es: String, reason: String?)
    case latency(Double?)
    case problem(String)
    case overflow

    /// seg_id puede venir como número (pipeline) o texto (modo incremental).
    private static func segID(_ v: Any?) -> String? {
        if let s = v as? String { return s }
        if let n = v as? NSNumber { return n.stringValue }
        return nil
    }

    static func parse(_ json: String) -> EngineEvent? {
        guard let data = json.data(using: .utf8),
              let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let type = obj["type"] as? String else { return nil }
        let id = segID(obj["seg_id"])
        let from = segID(obj["from_seg_id"])
        let es = obj["text_es"] as? String ?? ""
        switch type {
        case "state": return .state(obj["state"] as? String ?? "idle", obj["error"] as? String)
        case "partial": return .partial(id: id, from: from, es: es)
        case "segment": return .segment(id: id, from: from, es: es, en: obj["text_en"] as? String ?? "")
        case "discarded": return .discarded(id: id, from: from, es: es, reason: obj["reason"] as? String)
        case "metrics": return .latency((obj["last_endpoint_to_audio_ms"] as? NSNumber)?.doubleValue)
        case "error": return .problem(obj["message"] as? String ?? "Error desconocido.")
        case "overflow": return .overflow
        default: return nil
        }
    }
}

final class EngineClient {
    let base: URL
    private var token = ""
    private let session: URLSession
    private let decoder = JSONDecoder()

    init(base: URL) {
        var comps = URLComponents(url: base, resolvingAgainstBaseURL: false)!
        comps.path = "/"
        comps.query = nil
        comps.fragment = nil
        self.base = comps.url!
        let cfg = URLSessionConfiguration.ephemeral
        cfg.timeoutIntervalForRequest = 60
        cfg.urlCache = nil
        session = URLSession(configuration: cfg)
    }

    func connect() async throws {
        let (data, resp) = try await session.data(from: base)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else { throw EngineError(message: "El motor no respondió.") }
        let html = String(decoding: data, as: UTF8.self)
        let pattern = #"name="ot-token" content="([^"]+)""#
        guard let re = try? NSRegularExpression(pattern: pattern),
              let m = re.firstMatch(in: html, range: NSRange(html.startIndex..., in: html)),
              let r = Range(m.range(at: 1), in: html) else {
            throw EngineError(message: "No se pudo abrir la sesión con el motor.")
        }
        token = String(html[r])
    }

    private func url(_ path: String) -> URL { URL(string: path, relativeTo: base)!.absoluteURL }

    @discardableResult
    private func send(_ method: String, _ path: String, body: [String: Any]? = nil) async throws -> Data {
        var req = URLRequest(url: url(path))
        req.httpMethod = method
        req.setValue(token, forHTTPHeaderField: "X-OT-Token")
        if let body {
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        } else if method == "POST" {
            req.httpBody = Data()
        }
        let (data, resp) = try await session.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        if code >= 400 {
            let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            let err = root?["error"] as? [String: Any]
            throw EngineError(message: (err?["message"] as? String) ?? "El motor respondió con el código \(code).")
        }
        return data
    }

    private func get<T: Decodable>(_ path: String, as type: T.Type) async throws -> T {
        try decoder.decode(T.self, from: try await send("GET", path))
    }

    func status() async throws -> EngineStatus { try await get("api/status", as: EngineStatus.self) }
    func devices() async throws -> DeviceList { try await get("api/devices", as: DeviceList.self) }
    func voices() async throws -> VoiceList { try await get("api/voices", as: VoiceList.self) }
    func virtualMic() async throws -> VirtualMicInfo { try await get("api/virtual-mic", as: VirtualMicInfo.self) }

    /// POST /api/devices/refresh reinicializa PortAudio (solo en reposo); la lista nueva se lee aparte, como hace la UI web.
    func refreshDevices() async throws -> DeviceList {
        _ = try await send("POST", "api/devices/refresh")
        return try await devices()
    }

    func configure(_ values: [String: Any]) async throws -> EngineStatus {
        try decoder.decode(EngineStatus.self, from: try await send("POST", "api/config", body: values))
    }

    func start() async throws { try await send("POST", "api/start") }
    func stop() async throws { try await send("POST", "api/stop") }
    func deleteHistory() async throws { try await send("DELETE", "api/history") }

    /// Flujo de eventos (SSE). Termina si se corta la conexión; quien lo use reconecta.
    func events() -> AsyncThrowingStream<EngineEvent, Error> {
        AsyncThrowingStream { cont in
            let task = Task {
                do {
                    var comps = URLComponents(url: self.url("api/events"), resolvingAgainstBaseURL: false)!
                    comps.queryItems = [URLQueryItem(name: "token", value: self.token)]
                    var req = URLRequest(url: comps.url!)
                    req.timeoutInterval = 120  // el servidor manda un latido cada 15 s
                    let (bytes, _) = try await self.session.bytes(for: req)
                    for try await line in bytes.lines {
                        guard line.hasPrefix("data: ") else { continue }
                        if let ev = EngineEvent.parse(String(line.dropFirst(6))) { cont.yield(ev) }
                    }
                    cont.finish()
                } catch {
                    cont.finish(throwing: error)
                }
            }
            cont.onTermination = { _ in task.cancel() }
        }
    }
}
