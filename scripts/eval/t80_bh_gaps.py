"""T-80 · ¿la prueba de QA mete huecos en BlackHole?

Reproduce los clips de artifacts/t67-clips por BlackHole 2ch igual que
deliverables/qa/tests/sesion_tienda.py (sd.play + sd.wait, ganancia 0.7) y, en
otro proceso, graba la entrada de BlackHole. Después alinea cada clip grabado
con el original y cuenta huecos: tramos de >= 20 ms en cero exacto dentro del
clip donde el original tiene señal, y cuánto más largo quedó el clip grabado.
Guarda también sd.get_status() del reproductor (output_underflow).

PLAY_LATENCY=0.2 abre el reproductor con 200 ms de búfer (por defecto: el bajo de PortAudio).
PLAY48=1 reproduce a 48 kHz (sin conversión de tasa en PortAudio/CoreAudio).
Uso: .venv/bin/python scripts/eval/t80_bh_gaps.py OUT_DIR [N_CLIPS]
"""
import json, os, subprocess, sys, time
import numpy as np, sounddevice as sd, soundfile as sf

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SR = 48000


def rec_main(out_wav, stop_file):
    dev = next(i for i, d in enumerate(sd.query_devices()) if "BlackHole 2ch" in d["name"] and d["max_input_channels"])
    chunks, flags = [], []
    def cb(indata, frames, t, status):
        if status:
            flags.append(str(status))
        chunks.append(indata[:, 0].copy())
    with sd.InputStream(device=dev, channels=2, samplerate=SR, dtype="float32", callback=cb):
        while not os.path.exists(stop_file):
            time.sleep(0.05)
    sf.write(out_wav, np.concatenate(chunks), SR)
    json.dump({"rec_status": flags}, open(out_wav + ".json", "w"))


def zero_runs(x, min_len):
    z = np.abs(x) < 1e-7
    runs, i, n = [], 0, len(z)
    d = np.diff(np.concatenate([[0], z.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 999
    stop = os.path.join(out, "stop")
    if os.path.exists(stop):
        os.remove(stop)
    rec_wav = os.path.join(out, "rec.wav")
    p = subprocess.Popen([sys.executable, __file__, "--rec", rec_wav, stop])
    time.sleep(1.5)
    manifest = [json.loads(l) for l in open(f"{R}/artifacts/t67-clips/manifest.jsonl")][:nmax]
    bh_out = next(i for i, d in enumerate(sd.query_devices()) if "BlackHole 2ch" in d["name"] and d["max_output_channels"])
    t0 = time.time(); plays = []
    for m in manifest:
        path = f"{R}/data/audio/{m['audio_path']}"
        x, sr = sf.read(path, dtype="float32"); x = np.stack([x, x], 1) if x.ndim == 1 else x
        if os.environ.get("PLAY48") == "1" and sr != SR:  # sin conversión de tasa en el reproductor
            n48 = int(len(x) * SR / sr)
            x = np.stack([np.interp(np.arange(n48) * sr / SR, np.arange(len(x)), x[:, c]) for c in range(x.shape[1])], 1).astype(np.float32)
            sr = SR
        tp = time.time() - t0
        kw = {"latency": float(os.environ["PLAY_LATENCY"])} if os.environ.get("PLAY_LATENCY") else {}
        sd.play(x * 0.7, sr, device=bh_out, **kw); sd.wait()
        st = str(sd.get_status())
        plays.append({"clip": os.path.basename(path), "sr": sr, "t_play": tp, "t_end": time.time() - t0, "play_status": st, "path": path})
        time.sleep(1.0)
    open(stop, "w").close(); p.wait()
    rec, _ = sf.read(rec_wav, dtype="float32")
    meta = json.load(open(rec_wav + ".json"))
    res = []
    for pl in plays:
        x, sr = sf.read(pl["path"], dtype="float32")
        x = x if x.ndim == 1 else x[:, 0]
        if sr != SR:
            x = np.interp(np.arange(int(len(x) * SR / sr)) * sr / SR, np.arange(len(x)), x).astype(np.float32)
        # ventana aproximada (el reloj de grabación arrancó ~1,5 s antes de t0)
        a = max(0, int((pl["t_play"] + 1.0) * SR)); b = min(len(rec), int((pl["t_end"] + 3.0) * SR))
        seg = rec[a:b]
        # alinear por el 1.er tramo con señal
        on_src = np.argmax(np.abs(x) > 1e-3); on_rec = np.argmax(np.abs(seg) > 1e-3 * 0.7)
        y = seg[on_rec:]
        src = x[on_src:]
        # largo con señal: hasta la última muestra no nula
        last_rec = len(y) - np.argmax(np.abs(y[::-1]) > 1e-4)
        last_src = len(src) - np.argmax(np.abs(src[::-1]) > 1e-4 / 0.7)
        gaps = []
        for s, e in zero_runs(y[:last_rec], int(0.02 * SR)):
            # si el original no estaba en silencio digital en el mismo tramo, es hueco
            ss = src[s:e] if e <= len(src) else src[s:]
            if len(ss) == 0 or np.max(np.abs(ss)) > 1e-4:
                gaps.append({"at_s": round(s / SR, 3), "ms": round((e - s) / SR * 1000, 1)})
        res.append({"clip": pl["clip"], "play_status": pl["play_status"], "extra_ms": round((last_rec - last_src) / SR * 1000, 1), "gaps": gaps})
        print(pl["clip"], pl["play_status"] or "-", "extra_ms", res[-1]["extra_ms"], "gaps", gaps, flush=True)
    json.dump({"rec_status": meta["rec_status"][:50], "n_rec_status": len(meta["rec_status"]), "clips": res,
               "load": os.getloadavg()}, open(os.path.join(out, "gaps.json"), "w"), indent=1)
    open(os.path.join(out, "DONE"), "w").close()


if __name__ == "__main__":
    if sys.argv[1] == "--rec":
        rec_main(sys.argv[2], sys.argv[3])
    else:
        main()
