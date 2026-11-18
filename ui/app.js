// Frontend de opentraductor, contrato v0 de Kestrel-7 (backend local).
// Solo habla con el servidor de este equipo (mismo origen). No graba audio.
// Los textos del servidor se insertan con textContent, nunca como HTML.

// Filas del texto en vivo, sin DOM (T-55, memory/decisiones/t55-contrato-eventos-frase.md).
// Una fila por seg_id, en orden de llegada: "partial" | "final" | "discarded".
// - from_seg_id reasigna la fila «live» al id definitivo (no crea otra).
// - Un texto vacío nunca pisa un texto ya mostrado.
// - discarded deja la fila visible, marcada, con su motivo.
// - trim() nunca borra la fila parcial en curso.
// Se prueba con `node --test tests/ui/` (module.exports al final).
var LiveRows = (function () {
  "use strict";

  function create(max) {
    const rows = []; // { id, es, raw, en, state, reason }
    let current = null; // seg_id del parcial en curso

    function find(id) {
      return rows.find((r) => r.id === id) || null;
    }

    // Fila de `id`; si no existe y viene from_seg_id con fila, esa fila pasa a llamarse `id`.
    function take(id, from) {
      let row = find(id);
      if (!row && from != null) {
        row = find(String(from));
        if (row) {
          if (current === row.id) current = id;
          row.id = id;
        }
      }
      return row;
    }

    function setText(row, text) {
      if (typeof text === "string" && text !== "") row.es = text;
    }

    function partial(ev) {
      const id = String(ev.seg_id);
      let row = take(id, ev.from_seg_id);
      if (!row) {
        if (!ev.text_es) return false; // nada que mostrar todavía
        row = { id, es: "", raw: undefined, en: "", state: "partial", reason: null };
        rows.push(row);
      }
      if (row.state !== "partial") return false; // ya cerrada: un parcial tardío no la reabre
      setText(row, ev.text_es);
      current = id;
      trim();
      return true;
    }

    function segment(ev) {
      const id = String(ev.seg_id);
      let row = take(id, ev.from_seg_id);
      if (!row) {
        row = { id, es: "", raw: undefined, en: "", state: "final", reason: null };
        rows.push(row);
      }
      row.state = "final";
      row.reason = null;
      setText(row, ev.text_es);
      row.raw = ev.text_es_raw;
      row.en = ev.text_en || "";
      if (current === id) current = null;
      trim();
      return true;
    }

    function discarded(ev) {
      const id = String(ev.seg_id);
      let row = take(id, ev.from_seg_id);
      if (!row) {
        if (!ev.text_es) return false; // sin fila y sin texto: no se muestra nada
        row = { id, es: "", raw: undefined, en: "", state: "discarded", reason: null };
        rows.push(row);
      }
      if (row.state === "final") return false; // ya tiene traducción
      row.state = "discarded";
      row.reason = ev.reason || null;
      setText(row, ev.text_es);
      if (current === id) current = null;
      trim();
      return true;
    }

    // Recorta las más viejas hasta `max`, salteando el parcial en curso.
    function trim() {
      while (rows.length > max) {
        const i = rows.findIndex((r) => r.id !== current);
        if (i < 0) break;
        rows.splice(i, 1);
      }
    }

    function clear() {
      rows.length = 0;
      current = null;
    }

    return { rows, partial, segment, discarded, clear, current: () => current };
  }

  return { create };
})();

if (typeof module !== "undefined" && module.exports) module.exports = LiveRows;

(function () {
  "use strict";
  if (typeof document === "undefined") return; // node --test: solo LiveRows

  const $ = (id) => document.getElementById(id);
  const MAX_LIVE = 6; // últimos segmentos por columna; el resto queda en el historial
  const ACTIVE = ["starting", "running", "stopping"];
  const NO_OUTPUT = "__none__"; // «solo subtítulos»: la traducción no suena (mismo valor que runtime/service.py)
  const SESSION_LABEL = {
    idle: "detenida",
    starting: "iniciando",
    running: "en marcha",
    stopping: "parando",
    error: "error",
  };
  const MODEL_LABEL = {
    not_loaded: "no cargado",
    loading: "cargando",
    ready: "listo",
    error: "error",
  };

  const el = {
    conn: $("conn"),
    sessionState: $("session-state"),
    start: $("start"),
    stop: $("stop"),
    models: $("models"),
    preflight: $("preflight"),
    input: $("input"),
    output: $("output"),
    monitorOn: $("monitor-on"),
    monitor: $("monitor"),
    monitorVol: $("monitor-vol"),
    monitorState: $("monitor-state"),
    refreshDevices: $("refresh-devices"),
    refreshStatus: $("refresh-status"),
    monitorHint: $("monitor-hint"),
    audioLatency: $("audio-latency"),
    voice: $("voice"),
    src: $("src"),
    tgt: $("tgt"),
    backlog: $("backlog"),
    overflow: $("overflow"),
    dropped: $("dropped"),
    pbDropped: $("pb-dropped"),
    outputHint: $("output-hint"),
    alerts: $("alerts"),
    history: $("history"),
    histEmpty: $("hist-empty"),
    exportTxt: $("export-txt"),
    exportJson: $("export-json"),
    clearHist: $("clear-hist"),
  };

  const tokenMeta = document.querySelector('meta[name="ot-token"]');
  const TOKEN = tokenMeta ? tokenMeta.getAttribute("content") || "" : "";

  // Estado mínimo de la UI. Se resincroniza con el servidor al reconectar.
  const ui = {
    ignored: [],  // controles que el modo del motor ignora: no se ofrecen
    connected: false,
    sessionState: "idle", // valor confirmado por el servidor
    lastActive: false, // true si la última señal era una sesión activa
    devicesOk: false,
    monitorAvailable: false,
    overflowCount: 0,
    outputs: [],
    inputs: [],
    refreshing: false,
    reselect: { input: false, output: false }, // tras un refresco: elección previa perdida o ambigua; queda VACÍA hasta que el usuario elija
    outputConfirmed: null, // id de la salida principal que el usuario confirmó como auriculares EN ESTA pestaña
    monitorConfirmed: null, // id del monitor que el usuario confirmó como auriculares EN ESTA pestaña
    monitorTouched: false, // el usuario tocó el monitor en esta pestaña: su preferencia manda sobre el valor guardado
    serverMonitorId: null, // monitor que el backend informó en la última sincronización (undefined = sin confirmar)
    lastMonitorPoll: 0,
    live: LiveRows.create(MAX_LIVE), // filas del texto en vivo por seg_id (T-55)
  };

  function apiError(data, status) {
    if (data && typeof data === "object" && data.error && data.error.message) {
      return data.error.message;
    }
    return `Error ${status}`;
  }

  async function api(path, options) {
    const opts = Object.assign({}, options || {});
    opts.headers = Object.assign({ "X-OT-Token": TOKEN }, opts.headers || {});
    if (opts.body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(opts.body);
    }
    let res;
    try {
      res = await fetch(path, opts);
    } catch (_) {
      throw new Error("No hay conexión con el backend.");
    }
    const text = await res.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch (_) {
        data = text;
      }
    }
    if (!res.ok) {
      throw new Error(apiError(data, res.status));
    }
    return data;
  }

  function showAlert(kind, text) {
    const div = document.createElement("div");
    div.className = "alert " + kind;
    div.textContent = text;
    el.alerts.prepend(div);
    while (el.alerts.children.length > 4) {
      el.alerts.removeChild(el.alerts.lastChild);
    }
  }

  function setConn(ok, text) {
    el.conn.textContent = text;
    el.conn.className = "conn " + (ok ? "ok" : "bad");
  }

  function fillSelect(select, items, emptyLabel, pickDefault) {
    const previous = select.value;
    select.textContent = "";
    if (!items || items.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = emptyLabel;
      select.appendChild(opt);
      return;
    }
    for (const it of items) {
      const opt = document.createElement("option");
      opt.value = it.id;
      // dispositivos traen name; voces traen label
      const text = it.name || it.label || it.id;
      opt.textContent = text + (it.virtual ? " (virtual)" : "");
      select.appendChild(opt);
    }
    if (items.some((it) => it.id === previous)) {
      select.value = previous;
    } else if (pickDefault) {
      const d = items.find((it) => it.default);
      if (d) select.value = d.id;
    }
  }

  function modelText(v) {
    if (v === null || v === undefined) return { label: "desconocido", detail: null };
    if (typeof v === "string") return { label: MODEL_LABEL[v] || v, detail: null };
    const label = MODEL_LABEL[v.state] || v.state || "desconocido";
    return { label, detail: v.detail || v.name || null };
  }

  function renderStatus(s) {
    const em = s.engine_mode || {};
    ui.ignored = Array.isArray(em.ignored_controls) ? em.ignored_controls : [];
    el.models.textContent = "";
    const models = s.models || {};
    for (const key of Object.keys(models)) {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = key.toUpperCase() + " ";
      const badge = document.createElement("span");
      const t = modelText(models[key]);
      badge.className = "badge " + (typeof models[key] === "object" ? models[key].state || "" : models[key] || "");
      badge.textContent = t.label;
      li.appendChild(name);
      li.appendChild(badge);
      if (t.detail) {
        const d = document.createElement("div");
        d.className = "muted small";
        d.textContent = t.detail;
        li.appendChild(d);
      }
      el.models.appendChild(li);
    }
    if (el.models.children.length === 0) {
      const li = document.createElement("li");
      li.className = "muted";
      li.textContent = "El backend no informó modelos.";
      el.models.appendChild(li);
    }

    el.preflight.textContent = "";
    for (const p of s.preflight || []) {
      const li = document.createElement("li");
      const badge = document.createElement("span");
      badge.className = "badge " + (p.ok ? "ready" : "error");
      badge.textContent = p.ok ? "ok" : "falla";
      const name = document.createElement("span");
      name.textContent = " " + (p.name || "comprobación");
      li.appendChild(badge);
      li.appendChild(name);
      if (p.detail) {
        const d = document.createElement("div");
        d.className = "muted small";
        d.textContent = p.detail;
        li.appendChild(d);
      }
      el.preflight.appendChild(li);
    }
    if (el.preflight.children.length === 0) {
      const li = document.createElement("li");
      li.className = "muted";
      li.textContent = "Sin comprobaciones informadas.";
      el.preflight.appendChild(li);
    }

    if (s.config) {
      if (s.config.input_id && !ui.reselect.input) el.input.value = s.config.input_id;
      if (s.config.output_id && !ui.reselect.output) el.output.value = s.config.output_id;
      // asignar .value por código no dispara 'change': el aviso debe recalcularse aquí
      updateOutputHint();
      if (s.config.voice) el.voice.value = s.config.voice;
      ui.serverMonitorId = s.config.monitor_id || null;
      if (ACTIVE.includes(s.state)) {
        // sesión en curso: se muestra el estado EFECTIVO tal cual lo informa el backend (nunca «apagado» si sigue sonando)
        el.monitorOn.checked = !!s.config.monitor_id;
        if (s.config.monitor_id) el.monitor.value = s.config.monitor_id;
      } else if (!ui.monitorTouched) {
        // carga inicial / nuevo inicio sin confirmación local de auriculares: la preferencia arranca apagada.
        // Es solo la preferencia de esta pestaña: no se envía nada al servidor hasta que el usuario inicie.
        el.monitorOn.checked = false;
        el.monitor.value = "";
      }
      if (typeof s.config.monitor_volume === "number") {
        el.monitorVol.value = String(Math.round(s.config.monitor_volume * 100));
      }
      if (s.config.audio_latency) el.audioLatency.value = s.config.audio_latency;
    }
    if (s.metrics) renderMetrics(s.metrics);
    applyState(s.state || "idle", s.error);
  }

  function renderMetrics(m) {
    if (typeof m.asr_backlog_s === "number") {
      el.backlog.textContent = Math.round(m.asr_backlog_s * 1000) + " ms";
    } else {
      el.backlog.textContent = "sin medir";
    }
    if (typeof m.overflow_count === "number") {
      ui.overflowCount = m.overflow_count;
    }
    el.overflow.textContent = String(ui.overflowCount);
    el.dropped.textContent = typeof m.dropped_ms === "number" ? Math.round(m.dropped_ms) + " ms" : "sin medir";
    // Reproducción aparte: solo el reproductor principal, el monitor no se suma.
    el.pbDropped.textContent =
      typeof m.playback_dropped_ms === "number" ? Math.round(m.playback_dropped_ms) + " ms" : "sin medir";
  }

  // Aviso según la salida elegida. Sin salida virtual no es un error: es el modo intérprete (la voz suena aquí).
  function updateOutputHint() {
    const chosen = (ui.outputs || []).find((it) => it.id === el.output.value);
    const inp = selectedInput();
    if (el.output.value === NO_OUTPUT) {
      el.outputHint.textContent = "Solo subtítulos: la traducción aparece en pantalla y no suena.";
      el.outputHint.className = "small";
    } else if (!chosen) {
      el.outputHint.textContent = "";
      el.outputHint.className = "small";
    } else if (chosen.virtual) {
      el.outputHint.textContent = "Salida virtual: la traducción llegará a la app de reuniones si la elegís allí como micrófono.";
      el.outputHint.className = "small ok";
    } else if (inp && !inp.virtual) {
      el.outputHint.textContent =
        "Modo intérprete: la traducción suena en esta salida. Usá auriculares: con parlantes, el micrófono la capta y la traduce otra vez. " +
        "Para que llegue a Meet, Zoom o Discord hace falta una salida virtual (BlackHole).";
      el.outputHint.className = "small warn";
    } else {
      el.outputHint.textContent = "Modo intérprete: la traducción suena en esta salida. Para que llegue a una reunión hace falta una salida virtual (BlackHole).";
      el.outputHint.className = "small";
    }
  }

  // Eco de la salida principal: micrófono real + salida que no es virtual ni «solo subtítulos».
  function outputFeedbackPossible() {
    const out = (ui.outputs || []).find((it) => it.id === el.output.value);
    const inp = selectedInput();
    return !!(out && inp && !out.virtual && !inp.virtual);
  }

  function ensureOutputConfirmed() {
    if (!outputFeedbackPossible()) return true;
    if (ui.outputConfirmed === el.output.value) return true;
    const name = deviceName(el.output.value);
    const looks = SPEAKER_RE.test(name) ? "El nombre «" + name + "» parece de parlantes. " : "";
    const text =
      looks +
      "Si la traducción suena por parlantes, el micrófono la capta y el sistema la traduce otra vez (puede inventar frases). " +
      "¿Confirmás que «" + name + "» son auriculares? Si no, aceptá «solo subtítulos» desde la lista de salidas.";
    if (window.confirm(text)) {
      ui.outputConfirmed = el.output.value;
      return true;
    }
    return false;
  }

  // ---- Monitor y realimentación. La heurística de nombre SOLO cambia el texto del aviso: nunca permite ni prohíbe nada.
  const SPEAKER_RE = /speaker|altavoz|altavoces|parlante|built-in|incorporad|macbook|imac|display/i;

  function selectedMonitor() {
    return (ui.outputs || []).find((it) => it.id === el.monitor.value) || null;
  }

  function selectedInput() {
    return (ui.inputs || []).find((it) => it.id === el.input.value) || null;
  }

  function deviceName(id) {
    const d = (ui.outputs || []).find((it) => it.id === id);
    return d ? d.name || d.id : id;
  }

  // Hay camino acústico posible: micrófono real y monitor que no es virtual.
  function monitorFeedbackPossible() {
    const mon = selectedMonitor();
    const inp = selectedInput();
    return !!(el.monitorOn.checked && mon && inp && !inp.virtual && !mon.virtual);
  }

  function confirmHeadphones() {
    const mon = selectedMonitor();
    if (!mon) return false;
    const name = mon.name || mon.id;
    const looks = SPEAKER_RE.test(name) ? "El nombre «" + name + "» parece de parlantes. " : "";
    const text =
      looks +
      "Si el monitor suena por parlantes, el micrófono capta la traducción y el sistema la traduce otra vez (puede inventar frases). " +
      "¿Confirmás que «" + name + "» son auriculares?";
    if (window.confirm(text)) {
      ui.monitorConfirmed = mon.id;
      return true;
    }
    return false;
  }

  // true si no hace falta confirmar o ya se confirmó ESTE dispositivo en esta pestaña.
  function ensureMonitorConfirmed() {
    if (!monitorFeedbackPossible()) return true;
    if (ui.monitorConfirmed === el.monitor.value) return true;
    return confirmHeadphones();
  }

  // Consulta liviana del estado EFECTIVO del monitor en el backend (no toca los controles).
  function pollServerMonitor() {
    ui.lastMonitorPoll = Date.now();
    return api("/api/status")
      .then((s) => {
        ui.serverMonitorId = (s && s.config && s.config.monitor_id) || null;
        updateMonitorHint();
      })
      .catch(() => {});
  }

  // Dos líneas coherentes: lo que el backend hace AHORA y lo que se pedirá en el PRÓXIMO inicio.
  function updateMonitorHint() {
    const live = ACTIVE.includes(ui.sessionState);
    if (live) {
      el.monitorState.textContent =
        ui.serverMonitorId === undefined
          ? "Monitor ahora: sin confirmar (consultando al backend)"
          : "Monitor ahora: " + (ui.serverMonitorId ? "activo en " + deviceName(ui.serverMonitorId) : "apagado");
    } else {
      const mon = selectedMonitor();
      el.monitorState.textContent =
        "Próximo inicio: monitor " + (el.monitorOn.checked && mon ? "activado en " + (mon.name || mon.id) : "apagado");
    }
    let text = "";
    if (!live && el.monitorOn.checked) {
      if (!selectedMonitor()) text = "Elegí el dispositivo del monitor (recomendado: auriculares).";
      else if (monitorFeedbackPossible() && ui.monitorConfirmed !== el.monitor.value) {
        text = "Usá auriculares: con parlantes y micrófono del equipo, el micrófono capta la traducción.";
      }
    }
    el.monitorHint.textContent = text;
    el.monitorHint.className = "small" + (text ? " warn" : "");
  }

  // Sin conexión no se afirma "detenida" si la última señal era una sesión activa.
  function sessionLabel() {
    if (!ui.connected) {
      return ui.lastActive ? "sin conexión (estado no confirmado)" : "sin conexión";
    }
    return SESSION_LABEL[ui.sessionState] || ui.sessionState;
  }

  function refreshControls() {
    const active = ACTIVE.includes(ui.sessionState);
    const haveSelection = !!el.input.value && !!el.output.value && ui.devicesOk;
    const canStart =
      ui.connected && haveSelection && !ui.refreshing && (ui.sessionState === "idle" || ui.sessionState === "error");
    // Parar queda accesible si la sesión puede seguir activa, también sin conexión.
    const canStop = active || (!ui.connected && ui.lastActive);

    el.sessionState.textContent = sessionLabel();
    el.start.disabled = !canStart;
    el.stop.disabled = !canStop;

    const locked = !ui.connected || active || ui.refreshing; // durante el refresco no se tocan selects ni se inicia
    el.input.disabled = locked || !ui.devicesOk;
    el.output.disabled = locked || !ui.devicesOk;
    el.voice.disabled = locked || ui.ignored.includes("voice");
    el.monitorOn.disabled = locked || !ui.monitorAvailable;
    el.monitor.disabled = locked || !el.monitorOn.checked;
    el.monitorVol.disabled = locked || !el.monitorOn.checked;
    el.audioLatency.disabled = locked;
    el.refreshDevices.disabled = locked || ui.refreshing; // nunca con sesión en curso (el servidor también lo rechaza)
    updateMonitorHint();
  }

  function applyState(state, error) {
    ui.sessionState = state || "idle";
    if (!ACTIVE.includes(ui.sessionState) && !ui.monitorTouched) {
      el.monitorOn.checked = false; // terminó una sesión que esta pestaña no configuró: el próximo inicio vuelve a empezar apagado
      el.monitor.value = "";
    }
    ui.lastActive = ACTIVE.includes(ui.sessionState);
    if (state === "error" && error) {
      showAlert("error", "Sesión: " + error);
    }
    refreshControls();
  }

  async function loadStatus() {
    renderStatus(await api("/api/status"));
  }

  async function loadDevices() {
    const d = await api("/api/devices");
    ui.inputs = Array.isArray(d.inputs) ? d.inputs : [];
    fillSelect(el.input, d.inputs, "Sin entradas detectadas", true);
    const prevOutput = el.output.value;
    fillSelect(el.output, d.outputs, "Sin salidas detectadas", true);
    // «Solo subtítulos» siempre existe: sirve sin BlackHole y sin salidas de audio.
    const noVoice = document.createElement("option");
    noVoice.value = NO_OUTPUT;
    noVoice.textContent = "Sin voz (solo subtítulos)";
    if (!(d.outputs && d.outputs.length)) el.output.textContent = "";
    el.output.appendChild(noVoice);
    if (prevOutput === NO_OUTPUT || !el.output.value) el.output.value = NO_OUTPUT;
    ui.outputs = Array.isArray(d.outputs) ? d.outputs : [];
    updateOutputHint();
    const prevMonitor = el.monitor.value;
    fillSelect(el.monitor, d.outputs, "Sin salidas detectadas", false);
    if (ui.outputs.length > 0) {
      // sin dispositivo preseleccionado: el monitor nunca cae por defecto en el primero de la lista
      const ph = document.createElement("option");
      ph.value = "";
      ph.textContent = "Elegí tus auriculares…";
      el.monitor.insertBefore(ph, el.monitor.firstChild);
      el.monitor.value = ui.outputs.some((it) => it.id === prevMonitor) ? prevMonitor : "";
    }
    ui.devicesOk = Array.isArray(d.inputs) && d.inputs.length > 0;
    ui.monitorAvailable = ui.devicesOk && ui.outputs.length > 0;
    if (ui.reselect.input) el.input.value = ""; // no se cae al dispositivo «por defecto» sin que el usuario elija
    if (ui.reselect.output) el.output.value = "";
  }

  // Pide al servidor reinicializar PortAudio (solo ocioso) y vuelve a leer la lista SIN perder lo elegido: fillSelect conserva la
  // selección previa si sigue existiendo. Si un dispositivo elegido desapareció, se avisa en vez de cambiarlo en silencio.
  async function refreshDevices() {
    if (ACTIVE.includes(ui.sessionState) || ui.refreshing) return;
    const before = { input: el.input.value, output: el.output.value, monitor: el.monitor.value };
    ui.refreshing = true;
    el.refreshStatus.textContent = "Actualizando…";
    refreshControls();
    try {
      const r = await api("/api/devices/refresh", { method: "POST" });
      await loadDevices();
      if (ui.monitorConfirmed && !ui.outputs.some((it) => it.id === ui.monitorConfirmed)) ui.monitorConfirmed = null;
      const info = (r && r.refresh) || {};
      const added = Array.isArray(info.added) ? info.added : [];
      const removed = Array.isArray(info.removed) ? info.removed : [];
      el.refreshStatus.textContent =
        "Lista actualizada" + (added.length ? "; nuevos: " + added.join(", ") : "") + (removed.length ? "; quitados: " + removed.join(", ") : "") + ".";
      // Una elección previa que ya no existe, o que es AMBIGUA (otro dispositivo con el mismo nombre y dirección, sin UID para distinguirlos),
      // queda VACÍA: no se cae al «por defecto» ni se conserva por id. Iniciar queda bloqueado hasta que el usuario elija de nuevo.
      const gone = [];
      const pick = (label, sel, list, prev, key) => {
        if (!prev || prev === NO_OUTPUT) return;
        const d = list.find((it) => it.id === prev);
        if (!d || d.ambiguous) {
          sel.value = "";
          if (key) ui.reselect[key] = true;
          gone.push(label + (d ? " (ambigua: hay otro dispositivo con el mismo nombre y dirección)" : " (ya no está)"));
        }
      };
      pick("entrada", el.input, ui.inputs, before.input, "input");
      pick("salida", el.output, ui.outputs, before.output, "output");
      pick("monitor", el.monitor, ui.outputs, before.monitor, null);
      const miss = Array.isArray(r && r.config_missing) ? r.config_missing : [];
      const amb = Array.isArray(r && r.config_ambiguous) ? r.config_ambiguous : [];
      for (const k of miss.concat(amb)) {
        if (k === "input_id") { ui.reselect.input = true; el.input.value = ""; }
        if (k === "output_id") { ui.reselect.output = true; el.output.value = ""; }
      }
      if (miss.length || amb.length) {
        const cands = (r && r.config_candidates) || {};
        const parts = miss.map((k) => k + " (ya no existe" + (cands[k] && cands[k].length ? "; ¿" + cands[k].join(" / ") + "?" : "") + ")").concat(amb.map((k) => k + " (ambiguo)"));
        showAlert("warn", "Dispositivo guardado que hay que elegir de nuevo: " + parts.join("; ") + ". No se reasigna solo.");
      }
      if (gone.length) showAlert("warn", "Elegí de nuevo: " + gone.join("; ") + ". No se reasigna solo y no se puede iniciar hasta elegir.");
      updateOutputHint();
    } catch (err) {
      el.refreshStatus.textContent = "";
      showAlert("error", err.message);
    } finally {
      ui.refreshing = false;
      refreshControls();
    }
  }

  async function loadVoices() {
    const v = await api("/api/voices");
    const list = (v && v.voices) || [];
    fillSelect(el.voice, list, "Sin voces");
    if (list.length && !list.some((x) => x.id === el.voice.value)) {
      el.voice.value = list[0].id;
    }
  }

  // Historial: GET /api/history?limit=50 -> {items, total}. Sin borrado por texto:
  // el contrato solo ofrece borrar todo o por sesión.
  async function loadHistory() {
    const data = await api("/api/history?limit=50&offset=0");
    renderHistory((data && data.items) || [], (data && data.total) || 0);
  }

  function renderHistory(items, total) {
    el.history.textContent = "";
    el.histEmpty.hidden = items.length > 0;
    for (const h of items) {
      const li = document.createElement("li");
      const left = document.createElement("div");
      const meta = document.createElement("div");
      meta.className = "ts";
      meta.textContent = h.ts || "";
      const es = document.createElement("div");
      setSourceText(es, h.text_es || "", h.text_es_raw);
      left.appendChild(meta);
      left.appendChild(es);
      const en = document.createElement("div");
      en.textContent = h.text_en || "";
      li.appendChild(left);
      li.appendChild(en);
      el.history.appendChild(li);
    }
    if (total > items.length) {
      const li = document.createElement("li");
      li.className = "muted small";
      li.textContent = `Mostrando ${items.length} de ${total}. Exportá para ver todo.`;
      el.history.appendChild(li);
    }
  }

  async function clearHistory() {
    if (!window.confirm("¿Borrar todo el historial de textos?")) return;
    try {
      await api("/api/history", { method: "DELETE" });
      await loadHistory();
    } catch (err) {
      showAlert("error", err.message);
    }
  }

  // Descarga con header de token (un enlace directo no lo enviaría).
  async function exportHistory(format) {
    try {
      const res = await fetch("/api/history/export?format=" + encodeURIComponent(format), {
        headers: { "X-OT-Token": TOKEN },
      });
      if (!res.ok) {
        const text = await res.text();
        let data = null;
        try {
          data = JSON.parse(text);
        } catch (_) {
          data = null;
        }
        throw new Error(apiError(data, res.status));
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "opentraductor-historial." + format;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      showAlert("error", err.message);
    }
  }

  async function startSession() {
    if (!ui.connected || !el.input.value || !el.output.value || ui.refreshing) return;
    if (el.monitorOn.checked && !el.monitor.value) {
      showAlert("error", "Activaste el monitor pero no elegiste dispositivo. Elegí auriculares o desactivá el monitor.");
      return;
    }
    if (!ensureOutputConfirmed()) {
      showAlert("warn", "Salida sin confirmar: elegí auriculares, una salida virtual o «Sin voz (solo subtítulos)».");
      return;
    }
    if (!ensureMonitorConfirmed()) {
      showAlert("warn", "Monitor sin confirmar: desactivalo o confirmá que usás auriculares.");
      return;
    }
    el.start.disabled = true;
    try {
      await api("/api/config", {
        method: "POST",
        body: {
          input_id: el.input.value,
          output_id: el.output.value,
          monitor_id: el.monitorOn.checked && el.monitor.value ? el.monitor.value : null,
          monitor_volume: el.monitorOn.checked ? Number(el.monitorVol.value) / 100 : 0.5,
          ...(ui.ignored.includes("voice") ? {} : { voice: el.voice.value }),
        },
      });
      await api("/api/start", { method: "POST" });
      applyState("starting");
    } catch (err) {
      showAlert("error", err.message);
      refreshControls();
    }
  }

  async function stopSession() {
    el.stop.disabled = true;
    try {
      await api("/api/stop", { method: "POST" });
      applyState("stopping");
    } catch (err) {
      showAlert("error", err.message + " Podés reintentar al reconectar.");
      refreshControls();
    }
  }

  async function changeMonitorVolume() {
    if (!ACTIVE.includes(ui.sessionState) && ui.sessionState !== "running") return;
    try {
      await api("/api/config", {
        method: "POST",
        body: { monitor_volume: Number(el.monitorVol.value) / 100 },
      });
    } catch (err) {
      showAlert("error", err.message);
    }
  }

  // Solo con la sesión detenida; el servidor también lo exige (409 si está activa).
  async function changeAudioLatency() {
    try {
      await api("/api/config", {
        method: "POST",
        body: { audio_latency: el.audioLatency.value },
      });
    } catch (err) {
      showAlert("error", err.message);
    }
  }

  // Columnas en vivo (T-55): el estado vive en ui.live (LiveRows, una fila por seg_id) y se
  // redibuja entero en cada evento (son MAX_LIVE filas). La columna en inglés muestra las finales.
  const DISCARD_LABEL = {
    short: "muy corta, no se tradujo",
    empty: "no se entendió, no se tradujo",
    cancelled: "cancelada, no se tradujo",
  };

  // Texto de origen. Si el ASR original difiere del texto que usó la traducción, se muestra debajo.
  // Sin original guardado (registros antiguos o motor sin dato) no se muestra nada extra.
  function setSourceText(node, text, raw) {
    node.textContent = text;
    if (typeof raw === "string" && raw !== text) {
      const orig = document.createElement("div");
      orig.className = "muted small";
      orig.textContent = "ASR original: " + raw;
      node.appendChild(orig);
    }
  }

  function liveItem(list, text, kind) {
    const li = document.createElement("li");
    li.className = kind;
    li.textContent = text;
    list.appendChild(li);
    return li;
  }

  function renderLive() {
    el.src.replaceChildren();
    el.tgt.replaceChildren();
    for (const row of ui.live.rows) {
      if (row.state === "final") {
        setSourceText(liveItem(el.src, "", "final"), row.es, row.raw);
        liveItem(el.tgt, row.en, "final");
      } else if (row.state === "discarded") {
        const li = liveItem(el.src, row.es, "discarded");
        const why = document.createElement("div");
        why.className = "muted small";
        why.textContent = "Descartada: " + (DISCARD_LABEL[row.reason] || "no se tradujo");
        li.appendChild(why);
      } else {
        liveItem(el.src, row.es, "partial");
      }
    }
    el.src.scrollTop = el.src.scrollHeight;
    el.tgt.scrollTop = el.tgt.scrollHeight;
  }

  function onPartial(ev) {
    if (ui.live.partial(ev)) renderLive();
  }

  function onSegment(ev) {
    ui.live.segment(ev);
    renderLive();
    loadHistory().catch((err) => showAlert("error", err.message));
  }

  function onDiscarded(ev) {
    if (ui.live.discarded(ev)) renderLive();
  }

  function handleEvent(ev) {
    switch (ev.type) {
      case "state": {
        const wasLive = ACTIVE.includes(ui.sessionState);
        applyState(ev.state);
        if (ACTIVE.includes(ev.state)) {
          if (!wasLive) ui.serverMonitorId = undefined; // sesión nueva: no se afirma nada hasta consultar al backend
          updateMonitorHint();
          pollServerMonitor();
        }
        break;
      }
      case "partial":
        onPartial(ev);
        break;
      case "segment":
        onSegment(ev);
        break;
      case "discarded":
        onDiscarded(ev);
        break;
      case "overflow":
        ui.overflowCount += 1;
        el.overflow.textContent = String(ui.overflowCount);
        showAlert("warn", "Se descartó audio por desborde" + (ev.reason ? ": " + ev.reason : "."));
        break;
      case "error":
        showAlert("error", ev.message || "Error desconocido del backend.");
        break;
      case "metrics":
        renderMetrics(ev);
        if (ACTIVE.includes(ui.sessionState) && Date.now() - ui.lastMonitorPoll > 3000) pollServerMonitor();
        break;
      default:
        break;
    }
  }

  async function resync() {
    // Dispositivos y voces antes que el estado: así config.input_id/output_id/voice
    // encuentran sus opciones al aplicarse.
    await loadDevices();
    await loadVoices();
    await loadStatus();
    await loadHistory();
    refreshControls();
  }

  function connectEvents() {
    if (!window.EventSource) {
      setConn(false, "Este navegador no soporta eventos en vivo.");
      return;
    }
    // EventSource no envía headers: el token va en la query para SSE.
    const es = new EventSource("/api/events?token=" + encodeURIComponent(TOKEN));
    es.onopen = () => {
      ui.connected = true;
      setConn(true, "Conectado al backend local.");
      resync().catch((err) => showAlert("error", err.message));
    };
    es.onerror = () => {
      ui.connected = false;
      setConn(false, "Sin conexión con el backend. Reintentando…");
      refreshControls();
    };
    es.onmessage = (msg) => {
      let ev;
      try {
        ev = JSON.parse(msg.data);
      } catch (_) {
        return;
      }
      handleEvent(ev);
    };
  }

  function wire() {
    el.start.addEventListener("click", startSession);
    el.stop.addEventListener("click", stopSession);
    el.input.addEventListener("change", () => {
      ui.reselect.input = false; // elección explícita
      refreshControls();
    });
    el.output.addEventListener("change", () => {
      ui.reselect.output = false;
      updateOutputHint();
      refreshControls();
    });
    el.monitorOn.addEventListener("change", () => {
      ui.monitorTouched = true;
      if (el.monitorOn.checked && !ensureMonitorConfirmed()) el.monitorOn.checked = false;
      refreshControls();
    });
    el.refreshDevices.addEventListener("click", refreshDevices);
    el.monitor.addEventListener("change", () => {
      ui.monitorTouched = true;
      if (ui.monitorConfirmed !== el.monitor.value) ui.monitorConfirmed = null; // otro dispositivo: se revalida
      if (el.monitorOn.checked && !ensureMonitorConfirmed()) el.monitor.value = "";
      refreshControls();
    });
    el.monitorVol.addEventListener("change", changeMonitorVolume);
    el.audioLatency.addEventListener("change", changeAudioLatency);
    el.exportTxt.addEventListener("click", () => exportHistory("txt"));
    el.exportJson.addEventListener("click", () => exportHistory("json"));
    el.clearHist.addEventListener("click", clearHistory);
  }

  function init() {
    wire();
    refreshControls();
    if (!TOKEN) {
      setConn(false, "Falta el token de sesión. Abrí la UI desde el servidor local.");
      return;
    }
    connectEvents();
    resync().catch((err) => showAlert("error", err.message));
  }

  document.addEventListener("DOMContentLoaded", init);
})();
