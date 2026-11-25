#!/usr/bin/env python3
"""T-69: genera scripts/eval/sets/charlas_ctx_ar.jsonl, 5 charlas de 16 turnos en español rioplatense.

Cada charla es una conversación seguida (el MT recibe los turnos anteriores como contexto). Cada turno puede llevar
``terms``: términos cuya traducción depende del tema (plata/pila/cola/turno/guardia/sale...). Un término está bien si
``must`` (regex, sin distinguir mayúsculas) aparece en la traducción y ``bad`` (regex) no aparece.
Métrica: términos bien traducidos / términos totales. Los regex son generosos: aceptan cualquier traducción razonable.
Escrito ANTES de ver la salida de Qwen; cualquier ajuste posterior de regex se registra en artifacts/t69-charlas/AJUSTES.md.
Uso: python3 scripts/eval/t69_make_charlas.py
"""
import json
from pathlib import Path

T = lambda term, must, bad="": {"t": term, "must": must, "bad": bad}  # noqa: E731

CHARLAS = {}

CHARLAS["plata"] = ("plata y transacciones", [
    ("Hola, ¿cómo andás? Te escribo por la plata del alquiler.", [T("plata", "money|cash|payment|funds", "silver")]),
    ("Ya te hice la transferencia desde el home banking.", [T("transferencia", "transfer", "transference"), T("home banking", "online banking|home banking|banking app|bank app|bank website|banking site|bank's app")]),
    ("Fijate si te llegó, me dicen que puede tardar hasta cuarenta y ocho horas hábiles.", [T("hábiles", "business|working", "skill|able")]),
    ("No me llegó nada. ¿Me pasás el CBU de nuevo?", [T("CBU", "CBU|account number|bank account")]),
    ("Te mandé el alias, es juan punto perez punto mp.", [T("alias", "alias")]),
    ("Perfecto. Con el aguinaldo me alcanza para ponerme al día.", [T("aguinaldo", "bonus|aguinaldo|13th|thirteenth|year-end|christmas pay|holiday pay")]),
    ("El dólar blue está a mil quinientos, así que me conviene cambiar ahora.", [T("dólar blue", "blue")]),
    ("Yo prefiero pagar en cuotas sin interés, así no me ahorco a fin de mes.", [T("cuotas", "installment|instalment|payments"), T("sin interés", "interest-free|no interest|without interest|zero interest|0%"), T("me ahorco", "strapped|stretched|squeez|tight|struggl|choke|broke|burden|pinch|crunch|overextend|short|strain", "hang")]),
    ("Me sacaron una comisión de diez lucas por el giro.", [T("lucas", "thousand|10,000|10k|ten grand|10 grand|10K", "luca"), T("comisión", "fee|commission|charge")]),
    ("Es un afano, no pienso pagar eso.", [T("afano", "rip-off|ripoff|rip off|robbery|scam|steal|swindle|theft|con\\b|daylight", "afano")]),
    ("Si te sobra algo, dejame un adelanto y el resto lo arreglamos después.", [T("adelanto", "advance|down payment|deposit|partial|upfront|something up front")]),
    ("Estoy al palo este mes, no tengo un mango.", [T("al palo", "broke|tight|short|strapped|struggling|pinched|maxed|cash-strapped|hard up", "pole|stick"), T("un mango", "penny|dime|cent|buck|broke|no money|nothing|dollar|peso", "mango")]),
    ("Por eso me anoté en el plazo fijo, es lo único que rinde.", [T("plazo fijo", "fixed-term|fixed term|term deposit|CD\\b|certificate of deposit|time deposit|fixed deposit|fixed-rate|fixed rate|fixed-deposit", "fixed deadline|fixed period only")]),
    ("Mirá, si no pagás la factura antes del vencimiento te cobran punitorios.", [T("vencimiento", "due date|deadline|expir|maturity|due", "winning"), T("punitorios", "penalt|late fee|late charge|interest|fine|surcharge|late payment")]),
    ("Te dejo el comprobante en el mail así tenés constancia.", [T("comprobante", "receipt|proof|confirmation|voucher|statement")]),
    ("Listo, quedamos así. Cualquier cosa me avisás, ¿sale?", [T("¿sale?", "ok|okay|sound good|deal|alright|all right|agreed|sure|works|fine|got it", "\\bleave|exit|come out|go out")]),
])

CHARLAS["laburo"] = ("laburo con clientes", [
    ("Buenas, soy Laura, hablamos ayer por el laburo del sitio nuevo.", [T("laburo", "work|job|project", "laburo")]),
    ("Sí, ya te armé la cotización con todo lo que pidieron.", [T("cotización", "quote|quotation|estimate|proposal|pricing|bid", "stock")]),
    ("Son tres etapas, y cada una tiene su entregable.", [T("etapas", "phase|stage"), T("entregable", "deliverable")]),
    ("El anticipo es el cincuenta por ciento y el saldo contra entrega.", [T("anticipo", "advance|down payment|deposit|upfront|up-front|prepayment"), T("saldo", "balance|remaining|rest|remainder", "balance sheet"), T("contra entrega", "upon delivery|on delivery|against delivery|at delivery|COD|when .{0,20}deliver|delivery")]),
    ("Me parece caro, ¿no podés bajar un poco el precio?", [T("bajar el precio", "lower|reduce|drop|bring down|cut|come down|discount|knock|go down")]),
    ("Podemos sacar el módulo de reportes y así nos sale más barato.", [T("sacar el módulo", "remove|take out|drop|cut|leave out|take away|skip|eliminate|without", "get the module out"), T("nos sale más barato", "cheaper|less|cost|lower|affordable", "\\bleave|exit|come out")]),
    ("Dale, lo hablo con mi socio y te confirmo mañana.", [T("dale", "ok|okay|sure|alright|all right|sounds good|will do|go ahead|great|fine|deal|yes|let's do it|got it"), T("socio", "partner", "member|associate")]),
    ("Ojo que el cliente anterior nos dejó un quilombo con la facturación.", [T("quilombo", "mess|chaos|problem|trouble|disaster|headache|jumble|hassle", "quilombo"), T("facturación", "billing|invoic", "turnover")]),
    ("Necesito que me mandes la factura A con el CUIT de la empresa.", [T("factura", "invoice|bill"), T("CUIT", "CUIT|tax ID|tax number|tax identification|taxpayer|tax id|tax registration")]),
    ("Estamos con poco margen, nos están ajustando los tiempos.", [T("ajustando", "tighten|squeez|cut|tight|reduc|compress|shorten|limit|pressur", "adjust")]),
    ("Me bancás una semana más con el diseño, que estamos hasta las manos.", [T("me bancás", "bear with|give me|wait|hold on|hang on|hang in|patience|extend|another week|one more week|an extra week|can you give|cut me some slack|stand", "bank"), T("hasta las manos", "swamped|slammed|overwhelmed|buried|snowed|up to our|flat out|busy|jammed|hands full|overloaded|up to my", "to the hands|until the hands")]),
    ("No te preocupes, pasate por la oficina y lo vemos en persona.", [T("pasate", "stop by|come by|drop by|come (over|to|in|down)|swing by|visit|pass by|pop|stop in|drop in|come see|come around")]),
    ("Mandame el brief por mail así lo arranco hoy mismo.", [T("brief", "brief"), T("arranco", "start|begin|kick|get started|get going|work on|get on|dive|tackle")]),
    ("Se nos pasó la fecha de entrega, perdón, fue una fiaca generalizada.", [T("fiaca", "lazy|laziness|sluggish|slump|lethargy|not feeling|low energy|lack of motivation|couldn't be bothered|inertia|apath|laz|listless|drag|can't be bothered|unmotivated", "fiaca")]),
    ("Para compensarte te hago el mantenimiento del primer mes sin cargo.", [T("sin cargo", "free|no charge|at no cost|no extra|complimentary|without charge|for free|no cost|at no", ""), T("mantenimiento", "maintenance")]),
    ("Excelente, entonces arrancamos el lunes.", [T("arrancamos", "start|begin|kick|get started|get going|launch|commence")]),
])

CHARLAS["tech"] = ("tech / equipo de software", [
    ("Che, se cayó el servidor de producción y no sé por qué.", [T("se cayó", "down|crash|went down|is down|failed|fell over|outage|not responding|died|went offline", "\\bfell\\b|fell down")]),
    ("Fijate en los logs si hay algún error en la cola de mensajes.", [T("cola", "queue", "tail"), T("logs", "log")]),
    ("Hay un hilo que quedó colgado y no libera la memoria.", [T("hilo", "thread", "string|yarn|wire"), T("colgado", "hung|stuck|frozen|hanging|blocked|stalled|locked|dangling|deadlock|not responding", "hung up")]),
    ("Levantalo de nuevo y fijate si responde.", [T("levantalo", "start|restart|bring|spin|boot|launch|run|relaunch|fire|get it (up|back)|back up", "\\blift|pick (it )?up|raise")]),
    ("Ahora sí anda. Igual dejame revisar qué commit lo rompió.", [T("anda", "work|running|up\\b|fine|ok\\b|okay|good|functioning|back", "walk"), T("commit", "commit")]),
    ("Fue el último merge, alguien pisó mi cambio en la rama principal.", [T("pisó", "overwr|overr|stomp|clobber|trampl|wipe|replac|erase|stepp?ed on|overlap", "tread"), T("rama", "branch", "tree|limb|ramo")]),
    ("Hay que hacer un rollback y después un hotfix urgente.", [T("rollback", "roll ?back|revert|undo"), T("hotfix", "hot ?fix|patch|quick fix|fix")]),
    ("Subí el fix a la rama de desarrollo y abrí el pull request.", [T("subí", "push|upload|commit|put|merge|submit|deploy|load|post", "climb|went up"), T("pull request", "pull request|PR\\b")]),
    ("Bajate los cambios antes de arrancar, así no tenés conflictos.", [T("bajate", "pull|download|fetch|get|grab|sync|update", "get off|lower|go down|descend"), T("conflictos", "conflict")]),
    ("El pipeline tarda una eternidad, la pila de pruebas es enorme.", [T("pipeline", "pipeline", "pipe\\b"), T("pila", "stack|pile|suite|set|batch|lot|bunch|heap|number|bunch|ton", "batter")]),
    ("Habría que paralelizar los tests y cachear las dependencias.", [T("paralelizar", "parallel", "paraly"), T("cachear", "cache|caching")]),
    ("Si migramos a la nube bajamos el costo de infraestructura.", [T("nube", "cloud"), T("bajamos el costo", "reduce|cut|lower|drop|bring down|decrease|save|reduction")]),
    ("El front le pega a una API que devuelve un error cuatrocientos cuatro, no sé si es el token.", [T("le pega", "call|hit|request|make|quer|send|talk|ping|access|contact|interact", "beat|punch|slap|stick")]),
    ("Reiniciá el contenedor y limpiá el caché, eso suele arreglarlo.", [T("reiniciá", "restart|reboot|reset|relaunch"), T("contenedor", "container"), T("limpiá", "clear|clean|flush|purge|wipe|empty")]),
    ("Mañana hacemos el deploy a las seis de la tarde, así no molestamos a nadie.", [T("deploy", "deploy|release|rollout|roll out|push"), T("molestamos", "disturb|bother|annoy|disrupt|interfere|affect|impact|inconvenience|interrupt|trouble|get in anyone")]),
    ("Dale, avisame cuando esté listo y lo probamos en staging.", [T("lo probamos", "test|try|check|verify|run"), T("staging", "staging")]),
])

CHARLAS["salud"] = ("salud", [
    ("Buen día, quería sacar un turno con el cardiólogo.", [T("turno", "appointment|slot|booking|visit", "\\bturn\\b|shift"), T("cardiólogo", "cardiolog|heart (doctor|specialist)")]),
    ("Tengo obra social, pero no sé si cubre la consulta.", [T("obra social", "insurance|health plan|coverage|health coverage|medical plan|social security|social health|health care plan|HMO|health benefit", "social work|welfare"), T("cubre", "cover")]),
    ("Me dijeron que traiga la orden y el carnet de la obra social.", [T("orden", "referral|order|prescription|authorization|doctor's|request|form|note"), T("carnet", "card|ID|membership")]),
    ("Hace una semana que me duele el pecho cuando subo escaleras.", [T("pecho", "chest", "breast|bust"), T("subo", "climb|go up|walk up|take|going up|ascend|use", "upload|post")]),
    ("Ayer fui a la guardia porque me faltaba el aire.", [T("guardia", "ER\\b|emergency|urgent care|A&E|casualty|ED\\b|E\\.R", "guard|duty|watch"), T("me faltaba el aire", "short of breath|couldn't breathe|could not breathe|shortness of breath|out of breath|struggl|breath|difficulty breathing|trouble breathing|breathless|air")]),
    ("Me hicieron un electro y una placa de tórax, y salió todo bien.", [T("electro", "EKG|ECG|electrocardiogram|electro|cardiogram"), T("placa", "x-?ray|chest film|radiograph|imaging|scan", "plate|plaque|badge"), T("salió todo bien", "(normal|fine|clear|good|well|OK|okay)", "\\bleft|\\bleave")]),
    ("Igual el doctor me pidió análisis de sangre y una ecografía.", [T("análisis", "test|work|lab|blood"), T("ecografía", "ultrasound|sonogram|echo", "echograph")]),
    ("También me recetó pastillas para la presión y me dijo que me cuide con la sal.", [T("recetó", "prescri", "recipe"), T("pastillas", "pill|tablet|medication|meds|medicine", "pastille"), T("presión", "blood pressure|hypertension")]),
    ("La receta la tengo que presentar en la farmacia.", [T("receta", "prescription", "recipe"), T("farmacia", "pharmacy|drugstore|chemist")]),
    ("Mi mamá está internada hace tres días por una neumonía.", [T("internada", "hospitali|admitted|in the hospital|in hospital|inpatient|in the clinic", "intern\\b|interned"), T("neumonía", "pneumonia")]),
    ("Le dieron el alta pero tiene que hacer reposo y kinesiología.", [T("alta", "discharge|released|let her go|sent (her )?home|let go", "\\bhigh\\b|tall"), T("kinesiología", "physical therap|physiotherap|PT\\b|rehab|physio|kinesiolog")]),
    ("El médico de cabecera dice que no es grave, que es un virus.", [T("médico de cabecera", "primary care|family doctor|general practitioner|GP\\b|family physician|personal doctor|regular doctor|PCP|primary doctor|family practi|primary physician", "bedside|head of bed")]),
    ("Tomate la fiebre cada seis horas y si pasa de treinta y ocho andá a la guardia.", [T("tomate la fiebre", "temperature|fever|check"), T("andá a la guardia", "ER\\b|emergency|urgent|A&E|hospital", "guard\\b|duty")]),
    ("Tengo que ayunar doce horas antes del estudio.", [T("ayunar", "fast"), T("estudio", "test|scan|exam|procedure|study|screening|imaging|testing|lab|work", "")]),
    ("Me dieron turno para el jueves a las ocho, hay que llegar media hora antes.", [T("turno", "appointment|slot|booked|scheduled", "\\bturn\\b|shift"), T("media hora", "half an hour|30 minutes|half hour|thirty minutes|half-hour")]),
    ("Ojalá no sea nada. Me quedo más tranquila si lo ve el especialista.", [T("tranquila", "relieved|at ease|calm|reassured|peace of mind|better|feel more|comfortable|settled|rest easier|easy")]),
])

CHARLAS["viaje"] = ("viaje", [
    ("Hola, quiero reservar un pasaje a Bariloche para el quince de enero.", [T("pasaje", "ticket|flight|fare", "passage"), T("reservar", "book|reserve")]),
    ("Busco precios en temporada baja, temporada alta y también opciones en micro.", [T("temporada baja", "low season|off-season|off season|low-season|offseason"), T("micro", "bus|coach", "\\bmicro\\b")]),
    ("El vuelo sale a las seis, pero tiene una escala en Córdoba.", [T("sale", "depart|leave|take off|is at|is scheduled|is set|goes at|flight at|at six|at 6"), T("escala", "layover|stopover|stop|connection|connecting", "scale")]),
    ("Mejor uno directo, no quiero perder el vuelo por una conexión ajustada.", [T("directo", "direct|nonstop|non-stop|straight"), T("ajustada", "tight|short|narrow|close", "adjust")]),
    ("Con el pasaje me incluyen una valija despachada de veintitrés kilos.", [T("valija", "suitcase|bag|luggage|baggage", "valise"), T("despachada", "check")]),
    ("Llevá el DNI o el pasaporte, en migraciones te lo piden sí o sí.", [T("migraciones", "immigration|border|passport control", "\\bmigration"), T("sí o sí", "definitely|without fail|for sure|absolutely|no matter|a must|mandatory|require|always|compulsory|must|no exception|have to|need|ask|they'll")]),
    ("En la aduana te revisan la mochila, así que no lleves nada de comida.", [T("aduana", "customs")]),
    ("Reservé un hospedaje cerca del centro, con desayuno incluido.", [T("hospedaje", "accommodation|lodging|hotel|place to stay|stay|hostel|guesthouse|guest house|B&B|room|inn|hostal|lodge", "hosting")]),
    ("El alquiler del auto sale cuarenta dólares por día más el seguro.", [T("alquiler", "rental|rent|hire"), T("sale", "cost|is|run|come to|comes to|price|\\$|40", "\\bleaves?\\b|come(s)? out|goes out|exits?")]),
    ("Ojo con las rutas, hay mucho tránsito y los peajes son caros.", [T("tránsito", "traffic", "transit"), T("peajes", "toll")]),
    ("Prefiero ir en colectivo por la ciudad, es más barato que un taxi.", [T("colectivo", "bus", "collective")]),
    ("La excursión incluye traslado desde el hotel y guía en español.", [T("traslado", "transfer|pickup|pick-up|pick up|transportation|shuttle|transport|ride"), T("guía", "guide", "guidebook")]),
    ("Che, no te olvides de sacar un seguro de viaje y de cambiar plata.", [T("seguro de viaje", "travel insurance|trip insurance"), T("cambiar plata", "exchange|change (some )?money|currency|cash|money", "silver")]),
    ("Llegamos a la noche, así que hacemos el check-in y a dormir.", [T("check-in", "check"), T("a dormir", "sleep|bed|crash|sleeping|rest|nap")]),
    ("Estaba todo lleno, nos tuvimos que bancar dos horas de cola en el aeropuerto.", [T("cola", "line|queue", "tail"), T("bancar", "endure|put up|stand|bear|suffer|wait|sit|stuck|deal|stood|spend|spent|take|brave|sweat|handle")]),
    ("Igual valió la pena, el paisaje es una masa.", [T("una masa", "awesome|amazing|incredible|great|beautiful|stunning|fantastic|cool|gorgeous|spectacular|breathtaking|wonderful|killer|mind-blowing|epic|the best|dope|sick|fabulous|superb|unreal|phenomenal|wow", "\\bmass\\b|dough|pasta")]),
])


def main() -> int:
    out = Path(__file__).resolve().parent / "sets" / "charlas_ctx_ar.jsonl"
    nt = 0
    with open(out, "w") as f:
        for cid, (tema, turnos) in CHARLAS.items():
            assert 15 <= len(turnos) <= 20, (cid, len(turnos))
            f.write(json.dumps({"id": cid, "tema": tema, "turnos": [{"es": es, "terms": terms} for es, terms in turnos]},
                               ensure_ascii=False) + "\n")
            nt += sum(len(t) for _, t in turnos)
    print(f"{out}: {len(CHARLAS)} charlas, {sum(len(v[1]) for v in CHARLAS.values())} turnos, {nt} términos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
