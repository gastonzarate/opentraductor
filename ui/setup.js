// Pantallas de micrófono virtual (BlackHole) y licencia. Mismo origen, textos con textContent (nunca HTML).
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const tokenMeta = document.querySelector('meta[name="ot-token"]');
  const TOKEN = tokenMeta ? tokenMeta.getAttribute("content") || "" : "";

  async function api(path, options) {
    const opts = Object.assign({}, options || {});
    opts.headers = Object.assign({ "X-OT-Token": TOKEN }, opts.headers || {});
    if (opts.body !== undefined && typeof opts.body !== "string") {
      opts.body = JSON.stringify(opts.body);
      opts.headers["Content-Type"] = "application/json";
    }
    let res;
    try {
      res = await fetch(path, opts);
    } catch (e) {
      throw new Error("No hay conexión con la app local.");
    }
    let data = null;
    try {
      data = await res.json();
    } catch (e) {
      data = null;
    }
    if (!res.ok) {
      const msg = data && data.error && data.error.message ? data.error.message : `Error ${res.status}`;
      throw new Error(msg);
    }
    return data;
  }

  function setState(node, text, cls) {
    node.textContent = text;
    node.classList.remove("state-ok", "state-warn", "state-bad");
    if (cls) node.classList.add(cls);
  }

  // ------------------------------------------------------- micrófono virtual
  const vm = {
    state: $("vmic-state"),
    steps: $("vmic-steps"),
    recheck: $("vmic-recheck"),
    msg: $("vmic-msg"),
    note: $("vmic-note"),
  };

  function renderSteps(steps, url) {
    vm.steps.replaceChildren();
    for (const s of steps) {
      const li = document.createElement("li");
      const b = document.createElement("strong");
      b.textContent = s.title;
      li.appendChild(b);
      const p = document.createElement("div");
      p.textContent = s.detail;
      li.appendChild(p);
      if (s.command) {
        const c = document.createElement("code");
        c.textContent = s.command;
        li.appendChild(c);
      }
      if (s.url) {
        const a = document.createElement("a");
        a.href = s.url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = "Abrir página de descarga";
        const wrap = document.createElement("div");
        wrap.appendChild(a);
        li.appendChild(wrap);
      }
      vm.steps.appendChild(li);
    }
    vm.steps.hidden = steps.length === 0;
  }

  async function loadVmic(refresh) {
    vm.msg.textContent = "";
    vm.recheck.disabled = true;
    try {
      if (refresh) {
        try {
          await api("/api/devices/refresh", { method: "POST" });
        } catch (e) {
          vm.msg.textContent = e.message; // p. ej. sesión en curso: igual se muestra lo ya detectado
        }
      }
      const d = await api("/api/virtual-mic");
      vm.note.textContent = d.note || "";
      if (!d.audio_ok) {
        setState(vm.state, "No se pudo leer el audio del sistema. Cerrá y abrí la app.", "state-bad");
        renderSteps([], "");
      } else if (d.installed) {
        setState(vm.state, `Listo: se detectó ${d.device_name}. Elegilo como salida acá y como micrófono en tu reunión.`, "state-ok");
        renderSteps([], "");
      } else {
        setState(vm.state, `Falta ${d.device_name}. Para usar la traducción en Meet, Zoom o Discord, seguí estos pasos:`, "state-warn");
        renderSteps(d.steps || [], d.download_url);
      }
    } catch (e) {
      setState(vm.state, e.message, "state-bad");
    } finally {
      vm.recheck.disabled = false;
    }
  }

  vm.recheck.addEventListener("click", () => loadVmic(true));

  // ----------------------------------------------------------------- licencia
  const lc = {
    card: $("license-card"),
    state: $("license-state"),
    form: $("license-form"),
    key: $("license-key"),
    notice: $("license-notice"),
    btn: $("license-activate"),
    msg: $("license-msg"),
  };

  function renderLicense(s) {
    if (!s || s.state === "disabled") {
      lc.card.hidden = true;
      return;
    }
    lc.card.hidden = false;
    lc.notice.textContent = (s.network_notice || "") + (s.instance_name ? ` Nombre que se envía: «${s.instance_name}».` : "");
    if (s.state === "licensed") {
      setState(lc.state, `Licencia activada (${s.key_hint}). Funciona sin conexión a internet.`, "state-ok");
      lc.form.hidden = true;
    } else if (s.state === "trial") {
      const d = s.trial_days_left;
      setState(lc.state, `Prueba gratis: ${d === 1 ? "te queda 1 día" : `te quedan ${d} días`}. Si ya compraste, activá tu clave.`, "state-warn");
      lc.form.hidden = false;
    } else {
      setState(lc.state, "Terminó la prueba gratis de 7 días. Ingresá tu clave de licencia para seguir traduciendo.", "state-bad");
      lc.form.hidden = false;
    }
  }

  async function loadLicense() {
    try {
      renderLicense(await api("/api/license"));
    } catch (e) {
      lc.card.hidden = false;
      setState(lc.state, e.message, "state-bad");
    }
  }

  lc.form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    lc.msg.textContent = "Activando…";
    lc.msg.classList.remove("state-bad", "state-ok");
    lc.btn.disabled = true;
    try {
      const s = await api("/api/license/activate", { method: "POST", body: { license_key: lc.key.value } });
      lc.key.value = "";
      lc.msg.textContent = "Licencia activada.";
      lc.msg.classList.add("state-ok");
      renderLicense(s);
    } catch (e) {
      lc.msg.textContent = e.message;
      lc.msg.classList.add("state-bad");
    } finally {
      lc.btn.disabled = false;
    }
  });

  loadVmic(false);
  loadLicense();
})();
