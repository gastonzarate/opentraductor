"""T-74: set chico de control para GLOSSARY_EXT, armado por el equipo de medición (18/11).

Escrito sin mirar la salida de Qwen, el set del 17/11 ni el banco v1 y sin abrir QWEN_GLOSSARY_EXT. Solo conocía la
lista de lemas del mensaje del commit 63488b4. Temas distintos de v1 y del set t74: asado, consultorio,
oficina y trámites, cumpleaños familiar.
Cada término lleva un grupo en el archivo lateral charlas_ctx_ar_t74b.groups.json (el runner no lo lee):
  A = léxico rioplatense general que NO está en la lista de EXT (mide que EXT no estorbe).
  B = lema que SÍ está en EXT, en su sentido lunfardo (mide si suma).
  C = control de sentido literal de un lema de EXT (EXT no debe dispararse; debe seguir traduciéndose en sentido común).
Un término acierta si ``must`` aparece y ``bad`` no (igual que v1). Regex amplias, fijadas antes de correr nada.

Uso: python3 scripts/eval/t74_make_set_bench.py -> scripts/eval/sets/charlas_ctx_ar_t74b.jsonl (+ .groups.json)
"""
import json
import re
from pathlib import Path

NEVER = ""


def T(g, t, must, bad=NEVER):
    return {"g": g, "t": t, "must": must, "bad": bad}


CHARLAS = [
    {"id": "asado", "tema": "asado con amigos del barrio", "turnos": [
        {"es": "Dale, caé a eso de las ocho que ya prendo el fuego.", "terms": [
            T("A", "caé", r"show up|come (over|by|around)|get (here|there|over)|drop (by|in)|swing by|arrive|stop (by|in)|be there|head over|turn up|come", r"\bfall|\bfell")]},
        {"es": "Traé unas facturas para el mate, que mi viejo se queda dormido sin eso.", "terms": [
            T("A", "facturas", r"pastr|baked|sweet|medialuna|croissant|treats|cookies|bread|danish", r"invoice|\bbills?\b"),
            T("A", "mi viejo", r"\b(dad|father|old man|pops|papa)\b", "")]},
        {"es": "Ponele sal a la carne antes de que la pongas en la parrilla.", "terms": [
            T("C", "ponele sal", r"(put|add|sprinkle|season|rub|throw|salt)\b.*\bsalt|salt\b.*(on|the meat)|season", r"say salt|let'?s say|suppose|like salt|for example|say,")]},
        {"es": "Está lleno de pibes y la parrilla está al palo, no entra ni un chorizo más.", "terms": [
            T("A", "pibes", r"kids|guys|young|boys|lads|dudes|youngsters|children|youths|teens|people", r"pibe"),
            T("A", "al palo", r"full|packed|loaded|maxed|at full|crowded|completely|stuffed|jammed|going full|capacity|overflow|filled", r"\bstick|\bpole")]},
        {"es": "Ese amigo tuyo es un chamuyero, te vende cualquier cosa.", "terms": [
            T("A", "chamuyero", r"smooth[- ]talk|sweet[- ]talk|talker|bullshit|fast[- ]talk|con (man|artist)|charmer|gift of (the )?gab|silver[- ]tongue|\bBS\b|talks? a good game|salesman|schmooz", r"chamuy")]},
        {"es": "Ayer se armó un quilombo porque alguien mandó un mensaje trucho.", "terms": [
            T("A", "quilombo", r"mess|chaos|commotion|uproar|ruckus|fuss|drama|trouble|big deal|scene|brawl|hassle|disaster|uproar|storm", r"quilombo|brothel"),
            T("A", "trucho", r"fake|bogus|phony|fraud|counterfeit|forged|sketchy|dodgy|false|shady|scam", r"trucho")]},
        {"es": "Atenti que se acerca la policía, apagá la música.", "terms": [
            T("B", "atenti", r"watch out|heads[- ]up|careful|look out|be careful|alert|pay attention|attention|beware|be (on|aware)|stay (alert|sharp)|keep an eye|cops? (are|is)|police (are|is)", r"\batten(ti|d)\b")]},
        {"es": "Si pinta, nos juntamos de nuevo el domingo.", "terms": [
            T("B", "si pinta", r"if (it|that|you|we|they|i|there|the|someone|anyone|everyone)\b|if possible|maybe|we could|let'?s see|if .{0,20}(feel|want|work|come up|happen|show|turn)", r"paint")]},
    ]},
    {"id": "consultorio", "tema": "ir al médico por cansancio y presión", "turnos": [
        {"es": "Che, estoy reventado, hace dos semanas que duermo mal.", "terms": [
            T("A", "reventado", r"exhausted|wiped|worn out|drained|beat|wrecked|dead tired|shattered|burn(ed|t) out|completely tired|fried|spent|knackered|exhausting|really tired|so tired|very tired|bushed", r"burst|explod|blown")]},
        {"es": "Pedí turno con la doctora pero me dieron para dentro de un mes.", "terms": [
            T("A", "turno", r"appointment|slot|booking|time|visit|consultation|reserv", r"\bturn\b|\bshift")]},
        {"es": "La obra social me cubre la consulta pero no los estudios.", "terms": [
            T("A", "obra social", r"health (insurance|plan|coverage|care)|insurance|medical (plan|coverage|insurance)|union|HMO|coverage", r"social (work|project|welfare)|public work|charity"),
            T("A", "estudios", r"tests|exams?\b|scans|lab|work-?up|examinations|screening|imaging|procedures|diagnostic|bloodwork|testing|x-?rays|checks", r"stud(y|ies)")]},
        {"es": "Me tomó la presión y la tengo por las nubes.", "terms": [
            T("A", "por las nubes", r"sky-?high|through the roof|very high|\bhigh\b|elevated|off the charts|way up|skyrocket|up in the|really high|extremely high|dangerously", r"(in|by|for|through) the clouds")]},
        {"es": "Me recetó unas pastillas y me dijo que no me automedique.", "terms": [
            T("A", "automedique", r"self-?medicat|medicate (my|your)self|treat (my|your)self|medicat(e|ing) (on )?(my|your) own|take (anything|other|medic)|on my own|on your own|without (a )?(prescription|doctor)|medicat", r"automedic")]},
        {"es": "Dice que es el estrés, que me agarró una contractura en la espalda.", "terms": [
            T("A", "contractura", r"muscle (spasm|tension|knot|contracture|tightness)|knot|spasm|tight|contracture|cramp|strain|stiff|contraction|tense", r"\bcontract(s|ed)?\b")]},
        {"es": "Pero me mandó a bajar un cambio y a descansar un par de días.", "terms": [
            T("A", "bajar un cambio", r"slow down|ease up|take it easy|dial (it )?back|scale back|calm down|ease off|take a step back|relax|rest|cut back|pace myself|slow it down|lighten|lower the (pace|intensity)", r"lower (a |the )?gear|shift down|downshift|change down|reduce (a )?change|lower a change|down a gear")]},
        {"es": "Estoy hecho pelota, pero por lo menos no es nada grave.", "terms": [
            T("B", "hecho pelota", r"exhausted|wiped|worn out|drained|beat|wrecked|dead tired|shattered|burn(ed|t) out|a mess|trashed|destroyed|fried|spent|knackered|in bad shape|in pieces|a wreck|done for|rough shape|really tired|so tired|very tired|bushed", r"\bball\b|made (into )?a ball|turned into a ball")]},
    ]},
    {"id": "oficina", "tema": "trámites, plata y trabajo en la oficina", "turnos": [
        {"es": "Tengo que hacer un trámite en la oficina de impuestos y no me abre la página.", "terms": [
            T("A", "trámite", r"errand|paperwork|procedure|process|formalit|filing|bureaucrat|application|transaction|administrative|file|paper|task|registration|formality", r"\btrami")]},
        {"es": "Me pidieron el recibo de sueldo y otros papeles.", "terms": [
            T("A", "recibo de sueldo", r"pay ?(slip|stub|check|stubs)|payslip|salary (slip|statement|stub)|pay statement|wage (slip|statement)|pay (receipt|record)|income (proof|statement)|paycheck|pay documents|proof of income|salary (receipt|proof|certificate)", r"NEVER_MATCHES_X")]},
        {"es": "Me cayó el aguinaldo, pero igual llego justo a fin de mes.", "terms": [
            T("A", "aguinaldo", r"bonus|christmas|13th|thirteenth|year-end|mid-year|aguinaldo|extra (pay|salary|month)", r"NEVER_MATCHES_X"),
            T("A", "llego justo", r"barely|just (make|made|get|getting|manage|about)|tight|just in time|by a thread|make ends meet|cutting it close|just enough|scrape|by the skin|just reach|just reaching|just short|hardly|only just|just arrive|just get by", r"arrive (right|just) at")]},
        {"es": "Mi jefe es un mandamás y no delega nada.", "terms": [
            T("A", "mandamás", r"bossy|\bboss|big shot|control freak|micromanag|domineering|authoritarian|in charge|top dog|honcho|bigwig|high[- ]and[- ]mighty|commanding|dictator|calls the shots|overbearing|controlling", r"mandamas|command more")]},
        {"es": "Hoy hubo paro de colectivos y llegué hecho un trapo.", "terms": [
            T("A", "paro", r"strike|stoppage|walkout|work stoppage|shutdown|walk-out|industrial action", r"bus stop|unemployment|\bstop(ped)?\b(?!page)"),
            T("A", "hecho un trapo", r"exhausted|worn out|wiped|dead tired|a wreck|drained|shattered|beat|a mess|wrecked|trashed|drenched|totally spent|fried|knackered|destroyed|bushed|so tired|very tired", r"\brags?\b")]},
        {"es": "El cepo del auto lo trabé con la llave y no sale más.", "terms": [
            T("C", "cepo", r"clamp|boot|lock|vise|vice|wheel|trap|steering|snare|immobili|stock|club|bar|device|grip|hold", r"currency|exchange|foreign|restriction|capital controls|dollar")]},
        {"es": "Voy a pedir una tarjeta nueva, la otra me la bloquearon por un consumo raro.", "terms": [
            T("A", "consumo", r"purchase|charge|transaction|spending|expense|activity|payment|usage|use|buying|spend", r"consumption")]},
        {"es": "Y encima las cuotas sin interés ya se terminaron.", "terms": [
            T("A", "cuotas sin interés", r"interest[- ]free|installment|0 ?%|no[- ]interest|zero[- ]interest|without interest|payment plan|instalment|free of interest|interest[- ]less", r"quota")]},
    ]},
    {"id": "cumple", "tema": "cumpleaños de la abuela", "turnos": [
        {"es": "Mañana es el cumple de mi abuela y estoy a las corridas.", "terms": [
            T("A", "a las corridas", r"running around|rushing|in a rush|scrambling|running (late|errands)|in a hurry|hectic|frantic|swamped|racing|rushed|busy|pressed for time|crazy|chaos|on the run|running", r"")]},
        {"es": "Compré una torta en la panadería de la esquina, un lujo.", "terms": [
            T("A", "torta", r"cake", r"\btort|\bpie\b|sandwich"),
            T("A", "un lujo", r"great|fantastic|awesome|amazing|wonderful|fabulous|excellent|delicious|gorgeous|stunning|a treat|a delight|top|splendid|beautiful|superb|nice|lovely|perfect", "")]},
        {"es": "También traje una rosca de pan dulce para el café.", "terms": [
            T("C", "rosca de pan dulce", r"ring|wreath|coil|round|twist|braid|roll|loaf|sweet bread|ring[- ]shaped|bread|cake|pastry|bun|wheel|donut|doughnut|circular", r"intrigue|scheme|politic|plot|gossip|\bscrew|thread")]},
        {"es": "Mi hermana es una despistada, seguro se olvida de traer los vasos.", "terms": [
            T("A", "despistada", r"forgetful|scatterbrained|absent[- ]minded|airhead|flaky|spacey|ditzy|distracted|dizzy|careless|clueless|scatter-?brain|flighty|forget|spaced out|space cadet|head in the clouds|inattentive|disorganized", r"despist")]},
        {"es": "Mi tío siempre cuenta el mismo chiste y todos nos hacemos los giles.", "terms": [
            T("A", "hacerse los giles", r"pretend|act (like|as if|dumb|clueless|innocent)|play dumb|play along|feign|fake|fool|play it|acting|pretending|laugh along|go along", r"\bgil")]},
        {"es": "Mamá quiere sacar una foto de toda la familia con el coso ese del celular.", "terms": [
            T("B", "el coso", r"thing|gadget|device|doohickey|thingy|thingamajig|contraption|whatchamacallit|that (stick|tripod|selfie|holder|gizmo|thing)|gizmo|stick|tripod|whatsit", r"\bbody\b|\bcorpse|\bcoso\b")]},
        {"es": "Mi prima va a chusmear todo lo que hay en la mesa, ya la conozco.", "terms": [
            T("B", "chusmear", r"gossip|snoop|nos[ey]y?|pry|peek|sneak|check (out|everything)|look (at|through|around)|poke|sniff|nosing|nosey|gawk|rubberneck|busybod|eavesdrop|spy|stick (her|their) nose|checking", r"chusme")]},
        {"es": "No te hagas problema, igual hay un montón de comida para todos.", "terms": [
            T("B", "no te hagas problema", r"don'?t worry|no worries|don'?t sweat|no problem|never mind|don'?t stress|don'?t (make|be|get)|it'?s fine|it'?s okay|don'?t bother|no big deal|relax|take it easy|not to worry", r"make (a )?problem|do (a )?problem|create (a )?problem"),
            T("B", "es un montón", r"a lot|lots|plenty|tons?|loads?|heaps|so much|a ton|a bunch|much|enough|an? (huge|large|big) (amount|quantity|pile)|abundan|more than enough|mountain", r"\bpile of mountain|a mountain of mountains")]},
    ]},
]


def main() -> None:
    base = Path(__file__).resolve().parent / "sets"
    out = base / "charlas_ctx_ar_t74b.jsonl"
    n = 0
    groups = {}
    clean = []
    for c in CHARLAS:
        turnos = []
        for i, turn in enumerate(c["turnos"], 1):
            terms = []
            for t in turn["terms"]:
                bad = t["bad"] if t["bad"] != "NEVER_MATCHES_X" else ""
                re.compile(t["must"]); re.compile(bad or "x")
                groups[f"{c['id']}:t{i}:{t['t']}"] = t["g"]
                terms.append({"t": t["t"], "must": t["must"], "bad": bad})
                n += 1
            turnos.append({"es": turn["es"], "terms": terms})
        clean.append({"id": c["id"], "tema": c["tema"], "turnos": turnos})
    out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in clean), encoding="utf-8")
    (base / "charlas_ctx_ar_t74b.groups.json").write_text(json.dumps(groups, ensure_ascii=False, indent=1), encoding="utf-8")
    cnt = {g: list(groups.values()).count(g) for g in "ABC"}
    print(f"{out}: {len(clean)} charlas, {sum(len(c['turnos']) for c in clean)} turnos, {n} términos {cnt}")


if __name__ == "__main__":
    main()
