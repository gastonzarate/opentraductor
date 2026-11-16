// T-55 integración: la web (LiveRows de ui/app.js) con la secuencia que publica el runtime real.
// El fixture lo graba tests/integration/t55_dump_events.py. Correr: node --test tests/ui/
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const LiveRows = require("../../ui/app.js");
const { scenarios } = require(path.join(__dirname, "..", "fixtures", "t55_runtime_events.json"));

const MAX_LIVE = 6; // el mismo valor que usa la web

const state = (r) => (r.state === "discarded" ? `discarded:${r.reason || "-"}` : r.state);
const view = (live) => live.rows.map((r) => [r.id, state(r), r.es, r.en]);

for (const sc of scenarios) {
  test(`runtime → web: ${sc.name}`, () => {
    const live = LiveRows.create(MAX_LIVE);
    const shown = new Set(); // frases que ya se vieron en pantalla (siguiendo from_seg_id)
    sc.events.forEach((ev, i) => {
      if (ev.from_seg_id != null && shown.delete(String(ev.from_seg_id))) shown.add(String(ev.seg_id));
      live[ev.type](ev);
      if (ev.text_es) shown.add(String(ev.seg_id));
      for (const id of shown) {
        const row = live.rows.find((r) => r.id === id);
        assert.ok(row && row.es, `evento ${i} (${ev.type} ${ev.seg_id}): la frase ${id} desapareció`);
      }
    });
    assert.deepEqual(view(live), sc.expect);
  });
}
