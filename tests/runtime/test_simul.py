"""T-67 paso 2: local agreement y cortes de cláusula (piezas puras de runtime/simul.py)."""
from runtime.simul import ClauseCommitter, agreed_prefix, next_clause_end


def test_agreed_prefix_ignores_case_and_punct():
    assert agreed_prefix("el granizo destruyó toda la planta", "El granizo, destruyó toda la plantación") == \
        ["El", "granizo,", "destruyó", "toda", "la"]
    assert agreed_prefix("", "hola") == []


def test_clause_end_before_cue_with_min_words():
    w = "fuimos al centro con mi hermana pero no".split()
    assert next_clause_end(w, 0) == 6  # corta antes de "pero"
    assert next_clause_end("fui pero no".split(), 0) is None  # cláusula demasiado corta


def test_clause_end_pair_needs_both_words():
    w = "no llegamos a tiempo al tren o".split()
    assert next_clause_end(w, 0) is None
    assert next_clause_end(w + ["sea"], 0) == 6


def test_clause_end_after_comma_needs_next_word():
    w = "cuando llegamos a la casa,".split()
    assert next_clause_end(w, 0) is None
    assert next_clause_end(w + ["estaba"], 0) == 5


def test_committer_confirms_only_after_two_partials_agree():
    c = ClauseCommitter()
    assert c.update("fuimos al centro con mi hermana pero") == []  # primera hipótesis: nada estable
    assert c.update("fuimos al centro con mi hermana pero no había") == ["fuimos al centro con mi hermana"]
    assert c.update("fuimos al centro con mi hermana pero no había nadie") == []
    assert c.rest("fuimos al centro con mi hermana pero no había nadie") == "pero no había nadie"


def test_committer_rewrite_falls_back_to_full_sentence():
    c = ClauseCommitter()
    c.update("fuimos al centro con mi hermana pero")
    c.update("fuimos al centro con mi hermana pero no")
    assert c.update("fuimos al cine con mi hermana pero no") == []  # contradice lo confirmado
    assert c.rest("fuimos al cine con mi hermana pero no") is None


def test_committer_without_clauses_returns_final():
    c = ClauseCommitter()
    c.update("el granizo destruyó")
    c.update("el granizo destruyó la plantación")
    assert c.rest("el granizo destruyó la plantación") == "el granizo destruyó la plantación"
