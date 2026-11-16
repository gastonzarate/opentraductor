// T-55: filas del texto en vivo (ui/app.js, LiveRows). Correr: node --test tests/ui/
const test = require("node:test");
const assert = require("node:assert/strict");
const LiveRows = require("../../ui/app.js");

const view = (live) => live.rows.map((r) => [r.id, r.state, r.es, r.en]);

test("una fila por seg_id: el parcial se actualiza en su fila y el final la cierra", () => {
  const live = LiveRows.create(6);
  live.partial({ seg_id: 1, text_es: "hola" });
  live.partial({ seg_id: 1, text_es: "hola qué tal" });
  live.segment({ seg_id: 1, text_es: "Hola, ¿qué tal?", text_en: "Hi, how are you?" });
  assert.deepEqual(view(live), [["1", "final", "Hola, ¿qué tal?", "Hi, how are you?"]]);
});

test("from_seg_id reasigna la fila live al id definitivo (no crea otra)", () => {
  const live = LiveRows.create(6);
  live.partial({ seg_id: "s:live:0", text_es: "buenos días" });
  live.partial({ seg_id: "s:c1", from_seg_id: "s:live:0", text_es: "Buenos días, quisiera" });
  live.partial({ seg_id: "s:live:1", text_es: "una mesa" });
  live.segment({ seg_id: "s:c1", text_es: "Buenos días, quisiera", text_en: "Good morning, I'd like" });
  assert.deepEqual(view(live), [
    ["s:c1", "final", "Buenos días, quisiera", "Good morning, I'd like"],
    ["s:live:1", "partial", "una mesa", ""],
  ]);
  assert.equal(live.current(), "s:live:1");
});

test("texto vacío no pisa texto ya mostrado", () => {
  const live = LiveRows.create(6);
  live.partial({ seg_id: 3, text_es: "quisiera reservar" });
  live.partial({ seg_id: 3, text_es: "" });
  assert.deepEqual(view(live), [["3", "partial", "quisiera reservar", ""]]);
  live.segment({ seg_id: 3, text_es: "", text_en: "" });
  assert.equal(live.rows[0].es, "quisiera reservar");
});

test("parcial vacío sin fila no crea una fila en blanco", () => {
  const live = LiveRows.create(6);
  assert.equal(live.partial({ seg_id: 4, text_es: "" }), false);
  assert.equal(live.rows.length, 0);
});

test("discarded deja la fila visible y marcada con el motivo", () => {
  const live = LiveRows.create(6);
  live.partial({ seg_id: 5, text_es: "eh" });
  live.discarded({ seg_id: 5, text_es: "", reason: "short" });
  assert.deepEqual(live.rows.map((r) => [r.id, r.state, r.es, r.reason]), [["5", "discarded", "eh", "short"]]);
  assert.equal(live.current(), null);
});

test("discarded sin fila y sin texto no muestra nada", () => {
  const live = LiveRows.create(6);
  assert.equal(live.discarded({ seg_id: 6, text_es: "", reason: "empty" }), false);
  assert.equal(live.rows.length, 0);
});

test("discarded sobre una fila ya final no la cambia", () => {
  const live = LiveRows.create(6);
  live.segment({ seg_id: 7, text_es: "Sí", text_en: "Yes" });
  live.discarded({ seg_id: 7, text_es: "", reason: "cancelled" });
  assert.equal(live.rows[0].state, "final");
});

test("el recorte nunca borra la fila parcial en curso", () => {
  const live = LiveRows.create(3);
  live.partial({ seg_id: "p", text_es: "frase larga en curso" }); // la más vieja
  for (let i = 0; i < 5; i++) live.segment({ seg_id: i, text_es: "f" + i, text_en: "e" + i });
  live.partial({ seg_id: "p", text_es: "frase larga en curso todavía" });
  assert.equal(live.rows.length, 3);
  assert.deepEqual(live.rows[0].id, "p");
  assert.equal(live.rows[0].es, "frase larga en curso todavía");
  assert.deepEqual(live.rows.slice(1).map((r) => r.id), ["3", "4"]);
});

test("un parcial tardío no reabre una fila final", () => {
  const live = LiveRows.create(6);
  live.segment({ seg_id: 8, text_es: "Listo", text_en: "Done" });
  live.partial({ seg_id: 8, text_es: "lis" });
  assert.deepEqual(view(live), [["8", "final", "Listo", "Done"]]);
});

test("motor viejo: parciales de la siguiente frase no se pierden al cerrar la anterior", () => {
  // Pipeline baseline: seg_id por frase, sin from_seg_id ni discarded.
  const live = LiveRows.create(6);
  live.partial({ seg_id: 1, text_es: "primera" });
  live.partial({ seg_id: 2, text_es: "segunda en curso" });
  live.segment({ seg_id: 1, text_es: "Primera.", text_en: "First." });
  assert.deepEqual(view(live), [
    ["1", "final", "Primera.", "First."],
    ["2", "partial", "segunda en curso", ""],
  ]);
});
