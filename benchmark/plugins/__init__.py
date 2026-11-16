"""Plugins opcionales de evaluación (dependencias pesadas, import perezoso).

Todo plugin devuelve resultados etiquetados como automáticos. Ningún proxy se
presenta como MOS humano. Si falta dependencia o pesos: ``PluginUnavailable``.
"""


class PluginUnavailable(RuntimeError):
    """Dependencia, modelo o pesos no disponibles: no se inventa un puntaje."""
