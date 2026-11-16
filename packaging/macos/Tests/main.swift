import Foundation

// T-55: pruebas de Transcript (lo que hace AppModel.handle con los subtítulos) y de EngineEvent.parse.
// Esta Mac tiene solo Command Line Tools (sin XCTest): se compila con swiftc junto a Transcript.swift y
// EngineClient.swift y corre como ejecutable. Correr: packaging/macos/Tests/run_tests.sh (archivo main.swift: swiftc solo acepta código suelto ahí)

var failures = 0
var count = 0

func check(_ cond: Bool, _ what: String, line: Int = #line) {
    count += 1
    if !cond {
        failures += 1
        print("FALLA (línea \(line)): \(what)")
    }
}

func ev(_ json: String) -> EngineEvent {
    guard let e = EngineEvent.parse(json) else { fatalError("no parsea: \(json)") }
    return e
}

func feed(_ t: inout Transcript, _ events: [String]) {
    for j in events { t.apply(ev(j)) }
}

func view(_ t: Transcript) -> [String] {
    t.rows.map { r in
        let st: String
        switch r.state {
        case .partial: st = "partial"
        case .final: st = "final"
        case .discarded(let why): st = "discarded:\(why ?? "-")"
        }
        return "\(r.id)|\(st)|\(r.es)|\(r.en)"
    }
}

// .segment de la frase N no borra el parcial de la N+1 (el bug que vio la persona en la app).
do {
    var t = Transcript()
    feed(&t, [
        #"{"type":"partial","seg_id":1,"text_es":"primera"}"#,
        #"{"type":"partial","seg_id":2,"text_es":"segunda en curso"}"#,
        #"{"type":"segment","seg_id":1,"text_es":"Primera.","text_en":"First."}"#,
    ])
    check(view(t) == ["1|final|Primera.|First.", "2|partial|segunda en curso|"], "segment cierra solo su fila: \(view(t))")
}

// from_seg_id reasigna la fila live.
do {
    var t = Transcript()
    feed(&t, [
        #"{"type":"partial","seg_id":"s:live:0","text_es":"buenos días"}"#,
        #"{"type":"partial","seg_id":"s:c1","from_seg_id":"s:live:0","text_es":"Buenos días, quisiera"}"#,
        #"{"type":"partial","seg_id":"s:live:1","text_es":"una mesa"}"#,
        #"{"type":"segment","seg_id":"s:c1","text_es":"Buenos días, quisiera","text_en":"Good morning, I'd like"}"#,
    ])
    check(view(t) == ["s:c1|final|Buenos días, quisiera|Good morning, I'd like", "s:live:1|partial|una mesa|"],
          "from_seg_id reasigna: \(view(t))")
    check(t.current == "s:live:1", "parcial en curso = s:live:1")
}

// Texto vacío no pisa texto; parcial vacío sin fila no crea fila.
do {
    var t = Transcript()
    feed(&t, [
        #"{"type":"partial","seg_id":3,"text_es":"quisiera reservar"}"#,
        #"{"type":"partial","seg_id":3,"text_es":""}"#,
        #"{"type":"partial","seg_id":4,"text_es":""}"#,
    ])
    check(view(t) == ["3|partial|quisiera reservar|"], "vacío no pisa ni crea: \(view(t))")
    feed(&t, [#"{"type":"segment","seg_id":3,"text_es":"","text_en":""}"#])
    check(t.rows.first?.es == "quisiera reservar", "segment vacío no borra el español")
}

// discarded: visible y marcada; sin fila y sin texto, nada; sobre una final, no cambia.
do {
    var t = Transcript()
    feed(&t, [
        #"{"type":"partial","seg_id":5,"text_es":"eh"}"#,
        #"{"type":"discarded","seg_id":5,"text_es":"","reason":"short"}"#,
        #"{"type":"discarded","seg_id":6,"text_es":"","reason":"empty"}"#,
        #"{"type":"segment","seg_id":7,"text_es":"Sí","text_en":"Yes"}"#,
        #"{"type":"discarded","seg_id":7,"text_es":"","reason":"cancelled"}"#,
        #"{"type":"discarded","seg_id":8,"text_es":"una mesa","reason":"cancelled"}"#,
    ])
    check(view(t) == ["5|discarded:short|eh|", "7|final|Sí|Yes", "8|discarded:cancelled|una mesa|"],
          "discarded: \(view(t))")
    check(t.current == nil, "sin parcial en curso tras descartar")
}

// El recorte nunca borra el parcial en curso.
do {
    var t = Transcript(max: 3)
    t.apply(ev(#"{"type":"partial","seg_id":"p","text_es":"frase larga"}"#))
    for i in 0..<5 { t.apply(ev(#"{"type":"segment","seg_id":\#(i),"text_es":"f\#(i)","text_en":"e\#(i)"}"#)) }
    check(t.rows.map(\.id) == ["p", "3", "4"], "recorte respeta el parcial: \(t.rows.map(\.id))")
}

// Un parcial tardío no reabre una final.
do {
    var t = Transcript()
    feed(&t, [
        #"{"type":"segment","seg_id":9,"text_es":"Listo","text_en":"Done"}"#,
        #"{"type":"partial","seg_id":9,"text_es":"lis"}"#,
    ])
    check(view(t) == ["9|final|Listo|Done"], "parcial tardío: \(view(t))")
}

// Motor viejo (sin seg_id): una sola línea parcial que el segment vacía, como antes.
do {
    var t = Transcript()
    feed(&t, [
        #"{"type":"partial","text_es":"hola"}"#,
        #"{"type":"segment","text_es":"Hola.","text_en":"Hi."}"#,
    ])
    check(t.legacyPartial == "" && t.rows.count == 1 && t.rows[0].en == "Hi.", "motor viejo")
    feed(&t, [#"{"type":"partial","text_es":"chau"}"#, #"{"type":"state","state":"idle"}"#])
    check(t.legacyPartial == "", "idle borra el parcial del motor viejo")
    check(!t.isEmpty, "las finales quedan")
}

// parse: tipos desconocidos dan nil (clientes viejos los ignoran); seg_id número o texto.
check(EngineEvent.parse(#"{"type":"algo_nuevo"}"#) == nil, "tipo desconocido = nil")
if case .partial(let id, _, _) = ev(#"{"type":"partial","seg_id":12,"text_es":"x"}"#) {
    check(id == "12", "seg_id numérico")
} else { check(false, "partial") }

// Integración: la secuencia que publica el runtime real (tests/integration/t55_dump_events.py).
// Ninguna frase que ya se vio desaparece en ningún paso, y el estado final es el esperado.
if CommandLine.arguments.count > 1 {
    let url = URL(fileURLWithPath: CommandLine.arguments[1])
    guard let data = try? Data(contentsOf: url),
          let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
          let scenarios = root["scenarios"] as? [[String: Any]], !scenarios.isEmpty
    else { fatalError("no se pudo leer \(url.path)") }
    for sc in scenarios {
        let name = sc["name"] as? String ?? "?"
        let events = sc["events"] as? [[String: Any]] ?? []
        let expect = (sc["expect"] as? [[String]] ?? []).map { $0.joined(separator: "|") }
        var t = Transcript()
        var shown = Set<String>()
        for (i, obj) in events.enumerated() {
            let json = String(data: try! JSONSerialization.data(withJSONObject: obj), encoding: .utf8)!
            let id = obj["seg_id"].map { "\($0)" }
            if let from = obj["from_seg_id"].map({ "\($0)" }), shown.remove(from) != nil, let id { shown.insert(id) }
            t.apply(ev(json))
            if let id, let es = obj["text_es"] as? String, !es.isEmpty { shown.insert(id) }
            let lost = shown.filter { s in !t.rows.contains { $0.id == s && !$0.es.isEmpty } }
            check(lost.isEmpty, "runtime → app [\(name)] evento \(i): desaparecieron \(lost.sorted())")
        }
        check(view(t) == expect, "runtime → app [\(name)]: \(view(t)) != \(expect)")
    }
} else {
    print("AVISO: sin fixture del runtime (pasar tests/fixtures/t55_runtime_events.json)")
}

print(failures == 0 ? "OK: \(count) comprobaciones" : "\(failures) de \(count) fallaron")
exit(failures == 0 ? 0 : 1)
