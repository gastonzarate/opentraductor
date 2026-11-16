"""BUG-7: normalización previa a Marian ("120 mil" -> "120.000", una llamada por oración)."""
import pytest

from runtime import __main__ as cli
from runtime import components as K
from runtime.config import PipelineConfig


@pytest.mark.parametrize("src,out", [
    ("subió de 120 mil a 1,2 millones de dólares.", "subió de 120.000 a 1,2 millones de dólares."),
    ("unos 5 mil usuarios", "unos 5.000 usuarios"),
    ("1,2 millones", "1,2 millones"),          # no toca millones ni decimales
    ("mil gracias", "mil gracias"),            # sin cifra delante, no cambia
    ("1234 mil", "1234 mil"),                  # solo 1 a 3 cifras
    ("20 miles", "20 miles"),
])
def test_normalize_numbers(src, out):
    assert K.mt_normalize_numbers(src) == out


@pytest.mark.parametrize("src,out", [
    ("Nos vemos mañana a las 9, no a las 19.", "Nos vemos mañana a las 9, no a las 7 p. m."),
    ("a las 19", "a las 7 p. m."),
    ("a las 15:30 en punto", "a las 3:30 p. m. en punto"),
    ("desde las 13 hs.", "desde la 1 p. m."),
    ("a las 21 horas, ¿sí?", "a las 9 p. m., ¿sí?"),
    ("a las 19 personas", "a las 19 personas"),    # no es hora
    ("a las 9, no a las 12.", "a las 9, no a las 12."),  # 12 y menos: sin cambio
    ("las 24.", "las 24."),
    ("a las 190.", "a las 190."),
])
def test_normalize_hours(src, out):
    assert K.mt_normalize_numbers(src) == out


@pytest.mark.parametrize("src,parts", [
    ("¿Se puede pasar a la siguiente diapositiva? No veo el gráfico.",
     ["¿Se puede pasar a la siguiente diapositiva?", "No veo el gráfico."]),
    ("Publicamos la versión 2.0.0 ayer.", ["Publicamos la versión 2.0.0 ayer."]),
    ("Subió a 1,2 millones. ¡Bien!", ["Subió a 1,2 millones.", "¡Bien!"]),
    ("hola. chau", ["hola. chau"]),            # minúscula después: no corta
    ("  ", []),
])
def test_split_sentences(src, parts):
    assert K.mt_split_sentences(src) == parts


class _Fake(K.MarianCT2):
    def __init__(self, normalize):  # sin modelo: registra lo que recibe Marian
        self.normalize = normalize
        self.calls = []

    def _translate_one(self, text_es):
        self.calls.append(text_es)
        return f"<{text_es}>"


def test_translate_off_is_single_call_unchanged():
    m = _Fake(False)
    assert m.translate(" Sí. No 3 mil. ") == "<Sí. No 3 mil.>"
    assert m.calls == ["Sí. No 3 mil."]


def test_translate_on_splits_and_normalizes():
    m = _Fake(True)
    assert m.translate("Sí. No 3 mil.") == "<Sí.> <No 3.000.>"
    assert m.calls == ["Sí.", "No 3.000."]
    assert m.translate("") == "" and m.calls == ["Sí.", "No 3.000."]


def _cfg(tmp_path, *extra):
    a = cli.build_parser().parse_args(["run", "--db", str(tmp_path / "h.sqlite3"), *extra])
    svc, cfg = cli.build_service(a)
    svc.close()
    return cfg


def test_config_cli_and_env(tmp_path, monkeypatch):
    assert PipelineConfig().mt_normalize is False
    assert PipelineConfig.from_dict({"mt_normalize": True}).mt_normalize is True
    monkeypatch.delenv("OPENTRADUCTOR_MT_NORMALIZE", raising=False)
    assert _cfg(tmp_path).mt_normalize is False
    assert _cfg(tmp_path, "--mt-normalize", "on").mt_normalize is True
    monkeypatch.setenv("OPENTRADUCTOR_MT_NORMALIZE", "1")
    assert _cfg(tmp_path).mt_normalize is True
    assert _cfg(tmp_path, "--mt-normalize", "off").mt_normalize is False


def test_spec_fingerprint_includes_flag():
    from runtime.pipeline import Engine

    async def emit(*_):
        pass

    assert Engine(PipelineConfig(), emit)._spec_fp() != Engine(PipelineConfig(mt_normalize=True), emit)._spec_fp()
