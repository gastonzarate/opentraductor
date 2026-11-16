#!/usr/bin/env python3
"""[bench] p50/p95 de end_to_first_ms (titular) de cada carpeta de compare_configs.sh. Uso: ab_summary.py DIR..."""
import json, sys
sys.path.insert(0, "scripts/eval")
from gate import find_quantile, find_key  # noqa: E402
for d in sys.argv[1:]:
    data = json.load(open(f"{d}/run.json"))
    p50, p95 = find_quantile(data, "p50"), find_quantile(data, "p95")
    print(f"{d}: p50={p50[1] if p50 else None} p95={p95[1] if p95 else None} wer={(find_key(data,'wer') or (None,None))[1]}")
