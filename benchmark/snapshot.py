"""Snapshot reproducible de entorno, hardware, versiones y configuración."""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKAGES = ("numpy", "soundfile", "sacrebleu", "jiwer", "pytest")


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5
        ).stdout.strip() or None
    except Exception:
        return None


def hardware() -> dict[str, Any]:
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
        "hw_model": _sysctl("hw.model"),
        "ncpu": os.cpu_count(),
        "memsize_bytes": _sysctl("hw.memsize"),
    }


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def code_hash(package_dir: str | Path | None = None) -> str:
    """sha256 sobre el código del benchmark (el corpus no usa Git)."""
    root = Path(package_dir) if package_dir else Path(__file__).parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_snapshot() -> dict[str, Any]:
    """Carga del sistema (contención). La exclusividad se limita al equipo
    opentraductor: procesos ajenos pueden consumir CPU/GPU. Se registra la carga y
    los procesos más activos que NO son de esta corrida (solo nombre y %CPU) y se
    declara si hubo contención posible; nunca se afirma máquina en reposo."""
    import os
    import subprocess

    try:
        la = os.getloadavg()
    except OSError:
        la = None
    ncpu = os.cpu_count() or 1
    others: list[tuple[float, str]] = []
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,%cpu=,comm="], capture_output=True, text=True, timeout=5).stdout
        mine = {os.getpid(), os.getppid()}
        for line in out.splitlines():
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            pid, ppid, cpu, comm = int(parts[0]), int(parts[1]), float(parts[2]), parts[3]
            if pid in mine or ppid in mine:
                continue
            others.append((cpu, os.path.basename(comm)))
    except Exception:
        pass
    others.sort(reverse=True)
    busy = [(c, n) for c, n in others if c >= 50.0]
    contention = bool((la and la[0] / ncpu > 0.5) or busy)
    return {
        "loadavg_1_5_15": la,
        "ncpu": ncpu,
        "other_processes_over_50pct_cpu": [{"comm": n, "cpu_pct": c} for c, n in busy[:5]],
        "other_cpu_pct_total": round(sum(c for c, _ in others), 1) if others else None,
        "contention_possible": contention,
        "note": "exclusividad solo del equipo opentraductor; procesos ajenos no se detienen ni se afirma máquina en reposo",
    }


KEY_PACKAGES = (
    "mlx", "mlx-lm", "mlx-audio", "mlx-whisper", "parakeet-mlx", "ctranslate2", "torch", "torchaudio",
    "transformers", "huggingface-hub", "misaki", "kokoro", "faster-whisper", "utmosv2", "timm",
    "onnxruntime", "numpy", "scipy", "soundfile", "sacrebleu", "jiwer",
)
# Directorios de primer nivel que NO son código del candidato/runtime.
NON_RUNTIME_DIRS = {".venv", "venv", "data", "artifacts", "models", "tests", "docs", "benchmark",
                    "__pycache__", ".git", "node_modules", ".pytest_cache"}
MAX_FILES_LISTED = 2000


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def all_distributions() -> dict[str, str]:
    """Todos los paquetes instalados (nombre -> versión): el entorno completo."""
    out = {}
    for d in md.distributions():
        name = d.metadata["Name"]
        if name:
            out[name.lower()] = d.version
    return dict(sorted(out.items()))


def key_package_versions() -> dict[str, str | None]:
    allp = all_distributions()
    return {p: allp.get(p.lower()) for p in KEY_PACKAGES}


def tree_hash(root: Path, *, exclude_dirs: set[str] = frozenset(), top_level_only_dirs: bool = False) -> dict:
    """sha256 del código (.py/.toml/.json/.sh/.swift/.ts...) bajo ``root``."""
    exts = {".py", ".sh", ".toml", ".json", ".yaml", ".yml", ".swift", ".m", ".h", ".c", ".cpp", ".ts", ".js", ".md"}
    files: dict[str, str] = {}
    if not root.exists():
        return {"n_files": 0, "sha256": None, "files": {}}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix not in exts:
            continue
        rel = p.relative_to(root)
        if rel.parts[0] in exclude_dirs or "__pycache__" in rel.parts or any(x.startswith(".") and x not in (".",) for x in rel.parts[:-1]):
            continue
        files[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    h = hashlib.sha256()
    for k, v in files.items():
        h.update(k.encode())
        h.update(v.encode())
    return {"n_files": len(files), "sha256": h.hexdigest(),
            "files": files if len(files) <= MAX_FILES_LISTED else "omitido (demasiados archivos): ver sha256 agregado"}


def project_snapshot(root: str | Path | None = None) -> dict:
    """Código del runtime/candidato y archivos de bloqueo del proyecto.

    El benchmark se hashea aparte (``benchmark_code_sha256``). El runtime es todo
    directorio de primer nivel que no sea datos/artefactos/modelos/tests/docs.
    """
    root = Path(root) if root else project_root()
    out: dict = {"root": str(root), "lock_files": {}, "runtime_code": {}}
    for name in ("uv.lock", "pyproject.toml", ".python-version"):
        f = root / name
        out["lock_files"][name] = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None
    for d in sorted(root.iterdir()) if root.exists() else []:
        if d.is_dir() and d.name not in NON_RUNTIME_DIRS and not d.name.startswith("."):
            out["runtime_code"][d.name] = tree_hash(d)
    agg = hashlib.sha256()
    for k, v in out["runtime_code"].items():
        agg.update(k.encode())
        agg.update((v["sha256"] or "").encode())
    out["runtime_code_sha256"] = agg.hexdigest()
    return out


# ---- huellas de pesos (sin releer modelos enormes en cada corrida) ----

def _hash_cache_path() -> Path:
    import os

    return Path(os.environ.get("OPENTRADUCTOR_HASH_CACHE") or project_root() / "artifacts" / "hash-cache.json")


def file_fingerprint(path: str | Path, *, cache_path: str | Path | None = None) -> dict:
    """sha256 de un archivo con caché por (ruta real, tamaño, mtime). En cachés de
    Hugging Face un blob LFS se llama como su sha256: se usa sin releer."""
    import json
    import re

    p = Path(path)
    real = p.resolve()
    st = real.stat()
    if re.fullmatch(r"[0-9a-f]{64}", real.name) and real.parent.name == "blobs":
        return {"size": st.st_size, "sha256": real.name, "source": "hf_blob_name"}
    cp = Path(cache_path) if cache_path else _hash_cache_path()
    try:
        cache = json.loads(cp.read_text()) if cp.exists() else {}
    except Exception:
        cache = {}
    key = f"{real}|{st.st_size}|{st.st_mtime_ns}"
    if key in cache:
        return {"size": st.st_size, "sha256": cache[key], "source": "cache"}
    h = hashlib.sha256()
    with real.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    cache[key] = h.hexdigest()
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(cache))
    except OSError:
        pass
    return {"size": st.st_size, "sha256": h.hexdigest(), "source": "computed"}


def weights_fingerprint(path: str | Path, *, cache_path: str | Path | None = None) -> dict:
    """Huella de un archivo o directorio de pesos (p. ej. snapshot de HF).
    Incluye la revisión si la ruta es ``.../snapshots/<rev>``. Para adaptadores:
    devolver esto dentro de ``describe()``."""
    p = Path(path)
    out: dict = {"path": str(p)}
    if p.parent.name == "snapshots":
        out["hf_revision"] = p.name
        out["hf_repo"] = p.parent.parent.name.removeprefix("models--").replace("--", "/")
    if p.is_file():
        out["files"] = {p.name: file_fingerprint(p, cache_path=cache_path)}
    else:
        out["files"] = {str(f.relative_to(p)): file_fingerprint(f, cache_path=cache_path)
                        for f in sorted(p.rglob("*")) if f.is_file()}
    h = hashlib.sha256()
    for k, v in out["files"].items():
        h.update(k.encode())
        h.update(v["sha256"].encode())
    out["sha256_aggregate"] = h.hexdigest()
    return out


def environment_snapshot(root: str | Path | None = None) -> dict:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "hardware": hardware(),
        "packages": package_versions(),
        "key_packages": key_package_versions(),
        "all_packages": all_distributions(),
        "benchmark_code_sha256": code_hash(),
        "project": project_snapshot(root),
        "load_at_start": load_snapshot(),
    }
