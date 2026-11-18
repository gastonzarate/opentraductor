"""T-74: set chico NUEVO de charlas es-AR para validar el glosario rioplatense antes de mirar el set oficial del 17/11.

Escrito el 18/11 por @macdev sin mirar la salida de Qwen ni el set del 17/11. Temas distintos del banco v1
(mudanza, changas/freelance, cancha y finde). Mezcla términos que están en QWEN_GLOSSARY con otros que no
(bondi, chorros, birras, garca, calefón, heladera, buena onda, ojo de la cara, hacer precio, mi vieja) para ver
si el glosario ayuda sin tapar lo demás. Léxico de uso general (fuentes: Diccionario de la lengua de la Argentina,
AAL; Diccionario de americanismos, ASALE; Wikipedia «Lunfardo»). Regex ``must``/``bad`` amplios, escritos antes
de correr nada; un término acierta si ``must`` aparece y ``bad`` no (igual que el banco v1).

Uso: python3 scripts/eval/t74_make_set.py  ->  scripts/eval/sets/charlas_ctx_ar_t74.jsonl
"""
import json
import re
from pathlib import Path

N20 = r"20[,. ]?000|twenty thousand|20 ?(k|thousand|grand)"
N40 = r"40[,. ]?000|forty thousand|40 ?(k|thousand|grand)"


def T(t, must, bad=""):
    return {"t": t, "must": must, "bad": bad}


CHARLAS = [
    {"id": "mudanza", "tema": "mudanza a un depto compartido", "turnos": [
        {"es": "Che, ¿te bancás ayudarme con la mudanza el sábado?", "terms": [
            T("te bancás", r"up for|willing|can you|could you|would you|mind|able to|free to", r"\bbank")]},
        {"es": "Alquilé un flete pero me salió un ojo de la cara, veinte lucas.", "terms": [
            T("ojo de la cara", r"arm and a leg|fortune|a lot|expensive|a ton|a bundle|pretty penny", r"\beye"),
            T("lucas", N20, r"\blucas")]},
        {"es": "El chabón del flete es re piola, me hizo precio.", "terms": [
            T("chabón", r"\b(guy|dude|man|fellow|driver|mover)\b", r"chab"),
            T("re piola", r"(really|super|very|so|pretty|real) (nice|cool|great|chill|decent|friendly)", r"piola"),
            T("hizo precio", r"discount|deal|good price|cut me|lower|break|knocked|cheaper|better price|reduced", r"made (a |the |me a )?price")]},
        {"es": "Hay una bocha de cajas y la heladera pesa un montón.", "terms": [
            T("bocha", r"a lot of|lots of|tons? of|loads of|bunch of|many|plenty|heaps|piles? of|mountain", r"bocha|\bball"),
            T("heladera", r"fridge|refrigerator", r"ice cream")]},
        {"es": "Si no terminamos al toque, el portero nos va a hinchar.", "terms": [
            T("al toque", r"quick|right away|fast|soon|asap|in no time|immediately|in a hurry|on time", r"\btouch"),
            T("hinchar", r"\bbug|annoy|bother|nag|grief|hard time|on our case|pester|complain|yell|chew us|nuts", r"swell|inflat")]},
        {"es": "Mi vieja me prestó guita para la garantía.", "terms": [
            T("mi vieja", r"\b(mom|mother|mum|mama|ma)\b", r"old (lady|woman)"),
            T("guita", r"money|cash|funds", r"guita"),
            T("garantía", r"guarant|deposit|security|collateral|co-?sign", r"warrant")]},
        {"es": "El depto es chiquito pero está joya, tiene balcón.", "terms": [
            T("joya", r"great|perfect|awesome|lovely|beautiful|amazing|fantastic|wonderful|nice|gem|excellent|cute", r"jewel")]},
        {"es": "Lo malo es que el dueño es medio garca, no quiso arreglar el calefón.", "terms": [
            T("garca", r"crook|cheat|swindl|scam|rip-?off|shady|sleaz|jerk|asshole|dishonest|greedy|stingy|cheap|bastard|prick|scum|tight", r"garca"),
            T("calefón", r"water heater|boiler|heater|geyser", r"calef")]},
        {"es": "Igual zafamos, el contrato es por dos años y no está en negro.", "terms": [
            T("zafamos", r"we('re| are)? (fine|ok|okay|good|covered|safe)|we got (by|away|lucky|off|out)|we manag|we made it|we pulled|lucky|worked out|not (so )?bad|could be worse|we're set|dodged", r"\bzaf"),
            T("en negro", r"off the books|under the table|informal|undeclared|unregistered|illegal|cash.only|off the record|legit|legal|official|on the books", r"\bin (the )?black\b")]},
        {"es": "Después de la mudanza armamos un asado y unas birras, ¿de una?", "terms": [
            T("birras", r"beer", r"birra")]},
    ]},
    {"id": "changas", "tema": "changas y freelance de diseño", "turnos": [
        {"es": "Estoy haciendo changas de diseño porque en la agencia me rajaron.", "terms": [
            T("changas", r"odd jobs|side (gig|job|work|project)|gigs|freelanc|small jobs|on the side|side hustle|little jobs|one-off", r"chang"),
            T("rajaron", r"fired|let (me )?go|laid (me )?off|sacked|kicked (me )?out|canned|dismissed", r"split|crack|tore|ripped")]},
        {"es": "Posta, la agencia era un quilombo y el jefe un chanta.", "terms": [
            T("posta", r"honestly|seriously|for real|truth|really|no joke|i swear|truly|to be honest|no kidding|frankly", r"\bposta\b|\bpost\b"),
            T("quilombo", r"mess|chaos|disaster|nightmare|madhouse|shambles|circus|zoo|wreck|shitshow|clusterf", r"brothel|quilombo"),
            T("chanta", r"phony|phoney|fraud|\bcon\b|crook|liar|bullshit|\bbs\b|fake|charlatan|scam|hustler|swindler|blowhard|show-?off|poser|bluffer|smooth talker|windbag", r"\bchant|chanta")]},
        {"es": "Ahora cobro en dólares, unos quinientos verdes por mes.", "terms": [
            T("quinientos verdes", r"(500|five hundred) (us |u\.s\. )?(dollars|bucks|usd)|\$ ?500|usd ?500|us\$ ?500", r"green")]},
        {"es": "Los cambio al blue porque el oficial no me conviene.", "terms": [
            T("blue", r"black market|informal|parallel|unofficial|street|blue (dollar|rate|market|exchange)", r""),
            T("oficial", r"official", r"officer")]},
        {"es": "El laburo es lindo pero hay que ponerse las pilas con los plazos.", "terms": [
            T("laburo", r"\b(work|job|gig)\b", r"\blabou?r\b"),
            T("las pilas", r"step up|act together|get going|get moving|get serious|on top of|buckle down|hustle|get cracking|focus|on the ball|stay sharp|pick up the pace|work hard|put in|get my|get our|get your|get one's|keep up|stay on|sharp|pull (my|our|your)|get on it|kick it", r"batter|piles")]},
        {"es": "Ayer pifié un presupuesto y le cobré de menos a un cliente.", "terms": [
            T("pifié", r"messed up|screwed up|wrong|botched|blew|mistake|error|miscalculat|blunder|goofed|got .* off|undercharged|underestimat|bungled|fumbled|slipped", r"\bpif"),
            T("presupuesto", r"quote|estimate|budget|bid|proposal|pricing", r"presupuest")]},
        {"es": "¡Qué macana! Bueno, no es para hacer un drama, la próxima garpa más.", "terms": [
            T("macana", r"shame|bummer|pity|drag|blunder|mess|ugh|what a|too bad|bad luck|oops|damn|bad break", r"macana|\bclub\b"),
            T("garpa", r"\bpay|charge", r"garp")]},
        {"es": "Hay otro cliente que me manguea logos gratis todo el tiempo.", "terms": [
            T("manguea", r"\bbegs?\b|bums|mooch|asks me for|asking me for|keeps asking|scroung|wants (free )?logos|hits me up|requests|pester|bugs me for|cadg|freeload|expects", r"mangu")]},
        {"es": "Le dije que ni en pedo, que eso es laburar al pedo.", "terms": [
            T("ni en pedo", r"no way|not a chance|hell no|never|no chance|not happening|absolutely not|forget it|over my dead|not in a million|no fucking way|no damn way|not on your life", r"fart|drunk"),
            T("al pedo", r"for (nothing|free)|pointless|in vain|waste|useless|unpaid|without pay|no reason|nothing|worthless|free labor", r"fart|\bass\b|drunk")]},
        {"es": "Igual el chabón es buena onda, así que le hago la gamba con algo chico.", "terms": [
            T("buena onda", r"nice|good (guy|vibes?|person|people|dude|sort)|cool|friendly|good-?natured|decent|chill|easygoing|great guy|sweet|kind|good one|alright", r"good wave"),
            T("la gamba", r"help|back|favor|favour|hand|support|solid|throw|hook|cut (him|them) some slack|go along|play along|something small for him|do (him|it)", r"\bleg\b|gamba|shrimp")]},
    ]},
    {"id": "cancha", "tema": "ir a la cancha el finde", "turnos": [
        {"es": "¿Vamos a la cancha el domingo o te da fiaca?", "terms": [
            T("cancha", r"game|match|stadium|field|pitch|ground", r""),
            T("fiaca", r"lazy|laz|feel like|can't be bothered|not up for|too tired|not in the mood|bother|effort|don't want|up for it|lethargic|sluggish", r"fiac")]},
        {"es": "Las entradas salen una banda, cuarenta lucas cada una.", "terms": [
            T("salen", r"cost|\bare\b|go for|run|priced|sell for", r"\bleave|go out|come out|get out|exit"),
            T("una banda", r"a lot|fortune|a ton|so much|expensive|crazy|tons|an arm|loads|pricey|a bundle|insane|steep|a killing|a pile", r"\bband\b"),
            T("lucas", N40, r"\blucas")]},
        {"es": "Uf, qué garrón, este mes no tengo un mango.", "terms": [
            T("garrón", r"bummer|drag|sucks|pain|shame|pity|ugh|bad luck|too bad|lousy|unfair|rough|crap|damn|ugh|what a", r"garr"),
            T("un mango", r"broke|a (dime|penny|cent|nickel|peso|buck|single|red cent)|no money|any money|nothing|a thing|cash|short|tight|zero|skint", r"mango")]},
        {"es": "Tranqui, te banco la entrada y después me la devolvés.", "terms": [
            T("tranqui", r"relax|no worries|don't worry|chill|easy|it's (fine|ok|okay|cool|all good)|calm|no problem|all good|it's on me", r"tranqui"),
            T("te banco", r"cover|pay|spot|get (you|your)|front|lend|loan|treat|buy|handle|take care of|i've got|i got|on me", r"\bbank|bench")]},
        {"es": "Sos un groso, che. La otra vez el bondi tardó una hora.", "terms": [
            T("groso", r"legend|star|great|awesome|best|amazing|champ|king|rock|hero|the man|lifesaver|\bace\b|boss|beast|incredible|angel|saint|cool|genius|gem|good friend|pal|bro|sweetheart|the best", r"gross|\bfat\b|thick"),
            T("bondi", r"\bbus", r"bondi")]},
        {"es": "Llevemos camisetas originales, las truchas que venden afuera son un desastre.", "terms": [
            T("truchas", r"fake|knock-?off|counterfeit|bootleg|imitation|cop(y|ies)|replica|phony|bogus|cheap ones|rip-?off", r"trout")]},
        {"es": "Ojo con los chorros a la salida, hay que estar atentos.", "terms": [
            T("chorros", r"thie|pickpocket|robber|crook|mugger|criminal|burglar|purse snatcher|snatch|bandit", r"\bjets?\b|stream|squirt|spray")]},
        {"es": "Si ganamos va a ser zarpado, la gente se pone re manija.", "terms": [
            T("zarpado", r"awesome|amazing|incredible|insane|crazy|epic|wild|unreal|unbelievable|huge|great|fantastic|massive|sick|over the top|nuts|unforgettable|legendary|madness|bonkers|off the (hook|charts)", r"zarp|sail"),
            T("re manija", r"hyped|excited|pumped|fired up|wild|crazy|worked up|eager|thrilled|stoked|go nuts|go crazy|electric|frenzy|psyched|amped|riled|nuts|ecstatic|euphori|mad\b|go off|lose (it|their minds)|go bananas", r"handle|crank|manija")]},
        {"es": "Y si perdemos, el embole de volver en el bondi es tremendo.", "terms": [
            T("embole", r"drag|boring|bore|tedious|pain|miserable|bummer|annoying|depressing|dreadful|awful|downer|chore|ordeal|grind|sucks|misery|torture|nightmare|hassle", r"embol")]},
        {"es": "Bueno, dejá de flashear y comprá las entradas, ¿dale?", "terms": [
            T("flashear", r"daydream|imagin|overthink|fantasi[sz]|carried away|dream|tripping|trip\b|making things up|worrying|freaking|stressing|spinning|going on|rambl|dwell|fooling around|messing around|wasting time|talking|hallucinat|getting ahead|joking", r"flash"),
            T("dale", r"\b(ok|okay|alright|all right|deal|sure|right|cool|got it|yeah|yes|will you)\b", r"give it|give him|hit it")]},
    ]},
]


def main() -> None:
    out = Path(__file__).resolve().parent / "sets" / "charlas_ctx_ar_t74.jsonl"
    n = 0
    for c in CHARLAS:
        for turn in c["turnos"]:
            for t in turn["terms"]:
                re.compile(t["must"]); re.compile(t["bad"] or "x")
                n += 1
    out.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in CHARLAS), encoding="utf-8")
    print(f"{out}: {len(CHARLAS)} charlas, {sum(len(c['turnos']) for c in CHARLAS)} turnos, {n} términos")


if __name__ == "__main__":
    main()
