import Foundation

/// Subtítulos en pantalla (T-55, memory/decisiones/t55-contrato-eventos-frase.md). Sin UI: se prueba con
/// packaging/macos/Tests/run_tests.sh. Misma lógica que LiveRows en ui/app.js.
/// - Una fila por `seg_id`, en orden de llegada; `.segment` cierra solo su fila.
/// - `from_seg_id` reasigna la fila «live» al id definitivo.
/// - Un texto vacío nunca pisa un texto ya mostrado.
/// - `.discarded` deja la fila visible y marcada.
/// - El recorte nunca borra el parcial en curso.
/// Sin `seg_id` (motor viejo) queda como antes: una sola línea parcial que el `.segment` vacía.
struct Transcript {
    enum State: Equatable { case partial, final, discarded(String?) }

    struct Row: Identifiable, Equatable {
        var id: String
        var es: String
        var en: String
        var state: State
    }

    private(set) var rows: [Row] = []
    /// Parcial sin `seg_id` (motor viejo).
    private(set) var legacyPartial = ""
    private(set) var current: String?
    let max: Int

    init(max: Int = 200) { self.max = max }

    var isEmpty: Bool { rows.isEmpty && legacyPartial.isEmpty }

    mutating func clear() {
        rows.removeAll()
        legacyPartial = ""
        current = nil
    }

    /// Índice de la fila de `id`; si no existe y `from` tiene fila, esa fila pasa a llamarse `id`.
    private mutating func take(_ id: String, from: String?) -> Int? {
        if let i = rows.firstIndex(where: { $0.id == id }) { return i }
        guard let from, let i = rows.firstIndex(where: { $0.id == from }) else { return nil }
        rows[i].id = id
        if current == from { current = id }
        return i
    }

    mutating func partial(id: String?, from: String?, es: String) {
        guard let id else { legacyPartial = es; return }
        var i = take(id, from: from)
        if i == nil {
            guard !es.isEmpty else { return }
            rows.append(Row(id: id, es: "", en: "", state: .partial))
            i = rows.count - 1
        }
        guard let i, rows[i].state == .partial else { return }
        if !es.isEmpty { rows[i].es = es }
        current = id
        trim()
    }

    mutating func segment(id: String?, from: String?, es: String, en: String) {
        guard let id else {
            legacyPartial = ""
            rows.append(Row(id: UUID().uuidString, es: es, en: en, state: .final))
            trim()
            return
        }
        let i: Int
        if let found = take(id, from: from) {
            i = found
        } else {
            rows.append(Row(id: id, es: "", en: "", state: .final))
            i = rows.count - 1
        }
        rows[i].state = .final
        if !es.isEmpty { rows[i].es = es }
        rows[i].en = en
        if current == id { current = nil }
        trim()
    }

    mutating func discarded(id: String?, from: String?, es: String, reason: String?) {
        guard let id else { return }
        var i = take(id, from: from)
        if i == nil {
            guard !es.isEmpty else { return }
            rows.append(Row(id: id, es: "", en: "", state: .partial))
            i = rows.count - 1
        }
        guard let i, rows[i].state != .final else { return }
        rows[i].state = .discarded(reason)
        if !es.isEmpty { rows[i].es = es }
        if current == id { current = nil }
        trim()
    }

    private mutating func trim() {
        while rows.count > max, let i = rows.firstIndex(where: { $0.id != current }) {
            rows.remove(at: i)
        }
    }

    /// Lo que AppModel.handle hace con cada evento del motor. Devuelve false si el evento no es de subtítulos.
    @discardableResult
    mutating func apply(_ ev: EngineEvent) -> Bool {
        switch ev {
        case .partial(let id, let from, let es): partial(id: id, from: from, es: es)
        case .segment(let id, let from, let es, let en): segment(id: id, from: from, es: es, en: en)
        case .discarded(let id, let from, let es, let reason): discarded(id: id, from: from, es: es, reason: reason)
        case .state(let s, _):
            // Como antes: al detenerse se va el parcial del motor viejo. Las filas con seg_id quedan.
            if s == "idle" || s == "error" { legacyPartial = "" }
            return false
        default: return false
        }
        return true
    }

    /// Texto corto para una fila descartada.
    static func discardLabel(_ reason: String?) -> String {
        switch reason {
        case "short": return "Muy corta, no se tradujo"
        case "empty": return "No se entendió, no se tradujo"
        case "cancelled": return "Cancelada, no se tradujo"
        default: return "No se tradujo"
        }
    }
}
