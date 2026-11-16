"""Estadísticas de latencia: observado, censurado, fallo y no medible por separado.

Cada stream/fragmento cae en exactamente una clase (``Obs.status``):

* ``observed``: latencia medida (ms).
* ``timeout``: no hubo primer audio válido hasta un deadline conocido. Es
  censura por derecha real: latencia > ``value_ms`` (el deadline de ese stream).
* ``failure``: fallo operacional (error, cuelgue, salida silenciosa). No tiene
  latencia; no es censura. No entra en la distribución de latencia observada.
* ``anticipated``: salida antes del fin anotado, contenido sin verificar. No se
  penaliza ni se premia en la distribución observada; se publica aparte junto
  al lag. En el compuesto cuenta como "sin salida válida verificada".
* ``unmeasurable``: dato desconocido/inválido (sin fin anotado, sin
  correspondencia). Nunca se convierte en censura ni en fallo.

Salidas:

1. ``censoring_aware``: cuantiles de la latencia condicionada a salida
   operacional, con Kaplan-Meier (valido con deadlines heterogéneos). Un
   cuantil que cae en la zona censurada es ``None`` ("no identificable").
2. ``composite_time_to_valid_output``: fallos, timeouts y anticipados como +inf.
   NO es la distribución de latencia observada; es el tiempo hasta salida
   válida, con penalización. Solo se calcula si hay un deadline común >= todos
   los valores observados (única condición en que +inf conserva el orden); si
   no, es "no identificable".

IC: bootstrap por cluster (hablante o grabación) que re-muestrea clusters
completos con sus clases. ``low_sample`` usa el número de clusters
independientes, no el de streams: repetir 10 clips 10 veces no da p99.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

INF = float("inf")
STATUSES = ("observed", "timeout", "failure", "anticipated", "unmeasurable")


@dataclass(frozen=True)
class Obs:
    status: str
    value_ms: float | None = None  # observed: latencia; timeout: deadline
    cluster: str | None = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status inválido {self.status!r}")
        if self.status in ("observed", "timeout"):
            if self.value_ms is None or not math.isfinite(self.value_ms):
                raise ValueError(f"{self.status} requiere value_ms finito")
        elif self.value_ms is not None and not math.isfinite(self.value_ms):
            raise ValueError("value_ms no finito")


def _validate(q: float | None = None, n_boot: int | None = None, ci_level: float | None = None) -> None:
    if q is not None and not (0.0 < q <= 1.0):
        raise ValueError(f"q fuera de (0, 1]: {q}")
    if n_boot is not None and n_boot <= 0:
        raise ValueError("n_boot debe ser > 0")
    if ci_level is not None and not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level debe estar en (0, 1)")


def make_obs(
    values: Sequence[float | None],
    clusters: Sequence[str] | None = None,
    *,
    deadline_ms: float | None = None,
) -> list[Obs]:
    """Atajo: ``None`` = timeout al ``deadline_ms``. Valida longitudes."""
    if clusters is not None and len(clusters) != len(values):
        raise ValueError(f"len(clusters)={len(clusters)} != len(values)={len(values)}")
    out = []
    for i, v in enumerate(values):
        c = clusters[i] if clusters is not None else None
        if v is None:
            if deadline_ms is None:
                raise ValueError("valor None (timeout) requiere deadline_ms")
            out.append(Obs("timeout", deadline_ms, c))
        else:
            out.append(Obs("observed", float(v), c))
    return out


# --------------------------------------------------------------------------
# Cuantiles
# --------------------------------------------------------------------------


def nearest_rank(sorted_vals: Sequence[float], q: float) -> float:
    _validate(q=q)
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    return sorted_vals[max(1, math.ceil(q * n)) - 1]


def km_quantile(records: Iterable[Obs], q: float) -> float | None:
    """Cuantil Kaplan-Meier. ``observed`` = evento, ``timeout`` = censura.

    Otras clases se ignoran. ``None`` si el cuantil no es identificable.
    """
    _validate(q=q)
    pts = [(r.value_ms, 1 if r.status == "observed" else 0) for r in records if r.status in ("observed", "timeout")]
    if not pts:
        return None
    pts.sort(key=lambda p: (p[0], -p[1]))  # eventos antes que censuras en empates
    at_risk = len(pts)
    surv = 1.0
    i = 0
    while i < len(pts):
        t = pts[i][0]
        d = c = 0
        while i < len(pts) and pts[i][0] == t:
            d += pts[i][1]
            c += 1 - pts[i][1]
            i += 1
        if d:
            surv *= 1 - d / at_risk
            if 1 - surv >= q - 1e-12:
                return t
        at_risk -= d + c
    return None


def composite_values(records: Iterable[Obs]) -> tuple[list[float], dict]:
    """Valores del compuesto (+inf = sin salida válida). Excluye unmeasurable."""
    rs = list(records)
    vals = []
    for r in rs:
        if r.status == "observed":
            vals.append(r.value_ms)
        elif r.status in ("timeout", "failure", "anticipated"):
            vals.append(INF)
    max_obs = max((r.value_ms for r in rs if r.status == "observed"), default=None)
    deadlines = {r.value_ms for r in rs if r.status == "timeout"}
    ok = True
    reason = None
    if max_obs is not None and any(d < max_obs for d in deadlines):
        ok = False
        reason = "deadlines heterogéneos menores que algún valor observado: +inf no conserva el orden"
    elif len(deadlines) > 1:
        # deadlines distintos pero todos >= máximo observado: el orden se conserva
        pass
    return vals, {"identifiable_ordering": ok, "reason": reason}


def composite_quantile(records: Iterable[Obs], q: float) -> float | None:
    vals, info = composite_values(records)
    if not info["identifiable_ordering"] or not vals:
        return None
    est = nearest_rank(sorted(vals), q)
    return None if est == INF else est


def _boot(
    records: list[Obs],
    fn,
    q: float,
    n_boot: int,
    seed: int,
    ci_level: float,
) -> tuple[list[float | None], dict[str, list[Obs]]]:
    groups: dict[str, list[Obs]] = {}
    for i, r in enumerate(records):
        groups.setdefault(r.cluster if r.cluster is not None else f"__i{i}", []).append(r)
    gl = list(groups.values())
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        sample: list[Obs] = []
        for _g in range(len(gl)):
            sample.extend(gl[rng.randrange(len(gl))])
        v = fn(sample, q)
        boots.append(INF if v is None else v)
    boots.sort()
    a = (1 - ci_level) / 2
    lo = boots[max(0, math.floor(a * len(boots)))]
    hi = boots[min(len(boots) - 1, math.ceil((1 - a) * len(boots)) - 1)]
    return [None if lo == INF else lo, None if hi == INF else hi], groups


def quantile_report(
    records: Sequence[Obs],
    q: float,
    *,
    kind: str = "censoring_aware",
    n_boot: int = 1000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> dict:
    _validate(q=q, n_boot=n_boot, ci_level=ci_level)
    fn = km_quantile if kind == "censoring_aware" else composite_quantile
    used = [
        r for r in records
        if (r.status in ("observed", "timeout") if kind == "censoring_aware" else r.status != "unmeasurable")
    ]
    n_clusters_independent = len({r.cluster if r.cluster is not None else f"__i{i}" for i, r in enumerate(used)})
    need = math.ceil(1 / (1 - q)) if q < 1 else 10**9
    if not used:
        return {"q": q, "n": 0, "n_clusters": 0, "value": None, "identifiable": False,
                "ci": [None, None], "low_sample": True, "min_n_for_quantile": need}
    est = fn(used, q)
    ci, _ = _boot(used, fn, q, n_boot, seed, ci_level)
    return {
        "q": q,
        "n": len(used),
        "n_clusters": n_clusters_independent,
        "clusters_assumed_independent": all(r.cluster is None for r in used),
        "value": est,
        "identifiable": est is not None,
        "ci": ci,  # extremo None = sin cota (censura/fallos)
        "ci_level": ci_level,
        "n_boot": n_boot,
        # El tamaño efectivo es el número de clusters independientes.
        "low_sample": n_clusters_independent < need or len(used) < need,
        "min_n_for_quantile": need,
    }


def summarize_latency(
    records: Sequence[Obs],
    *,
    quantiles: Sequence[float] = (0.5, 0.95, 0.99),
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    _validate(n_boot=n_boot)
    counts = {s: sum(1 for r in records if r.status == s) for s in STATUSES}
    n = len(records)
    observed = [r.value_ms for r in records if r.status == "observed"]
    _, comp_info = composite_values(records)
    out = {
        "n_total": n,
        "counts": counts,
        "rates": {s: (c / n if n else None) for s, c in counts.items()},
        "rates_wilson95": {s: wilson_interval(c, n) for s, c in counts.items()},
        "observed_min_ms": min(observed) if observed else None,
        "observed_max_ms": max(observed) if observed else None,
        "timeout_deadlines_ms": sorted({r.value_ms for r in records if r.status == "timeout"}),
        "censoring_aware": {
            "label": "latencia condicionada a salida operacional (Kaplan-Meier; timeouts = censura por derecha; fallos/anticipados/no medibles excluidos y contados aparte)",
            "quantiles": {f"p{round(q * 100)}": quantile_report(records, q, kind="censoring_aware", n_boot=n_boot, seed=seed) for q in quantiles},
        },
        "composite_time_to_valid_output": {
            "label": "tiempo hasta salida válida con penalización (+inf para fallo/timeout/anticipado sin verificar); NO es latencia observada",
            "ordering_check": comp_info,
            "quantiles": {f"p{round(q * 100)}": quantile_report(records, q, kind="composite", n_boot=n_boot, seed=seed) for q in quantiles},
        },
    }
    return out


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """IC de Wilson para una tasa. Supone muestras independientes."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))
