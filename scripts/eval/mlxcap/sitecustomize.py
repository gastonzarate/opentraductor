# [bench] T-58: aplica mx.set_cache_limit en cada proceso de medición cuando OT_MLX_CACHE_LIMIT_MB está definida.
# El motor aún no fija el tope (T-51); esto reproduce la config de la tienda sin tocar runtime/.
import os
_mb = os.environ.get("OT_MLX_CACHE_LIMIT_MB")
if _mb:
    try:
        import mlx.core as mx
        mx.set_cache_limit(int(float(_mb) * 1e6))
    except Exception as e:  # sin mlx (p. ej. .venv): no aplica
        print(f"[mlxcap] no se aplicó el tope: {e}")
