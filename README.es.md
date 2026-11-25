# opentraductor (resumen en español)

Traductor de voz **local** de español argentino a inglés de EE. UU. para macOS con Apple Silicon. Hablás en español y la Mac lo dice en inglés un instante después. Todo corre en tu equipo: reconocimiento de voz, traducción y síntesis.

> **Estado: experimental.** Hoy la demora entre el fin de tu frase y el primer audio en inglés es de unos 0,57 segundos (mediana). No hay promesa de 300 ms. Documentación completa en inglés: [README.md](README.md).

## Qué hace

Micrófono → VAD (Silero) → reconocimiento (Parakeet o Moonshine) → traducción (Qwen3 o Marian) → voz (Kokoro) → micrófono virtual para Meet o Discord.

- Sin red después de bajar los modelos. Sin telemetría ni cuentas.
- No guarda audio, solo textos en un historial local que podés exportar o borrar.
- Interfaz web local en `127.0.0.1`.

## Requisitos

Mac con Apple Silicon (probado en M5 Pro, macOS 26), Python 3.12 y [uv](https://docs.astral.sh/uv/). Para reuniones, BlackHole 2ch (GPL-3.0, no incluido: lo instalás vos).

## Instalación

```bash
git clone https://github.com/gastonzarate/opentraductor.git
cd opentraductor
uv sync --extra runtime
uv run --extra runtime python -m runtime.setup    # descarga modelos (único paso con red)
uv run --extra runtime python -m runtime check
uv run --extra runtime python -m runtime run      # abre http://127.0.0.1:8877/
```

El perfil más rápido (Moonshine + Qwen3) y el micrófono virtual están explicados en el README en inglés.

## Resultados medidos

Mediana desde el fin de habla hasta el primer audio con el código de este repositorio: 572 ms, 64 corridas (8 audios x 8 rondas, A/B intercalado), en una Mac con otras tareas abiertas (carga media 3,6, hasta 5,9). Con la Mac ocupada hay picos de cerca de 1 segundo (p95 1039 ms en esa corrida). En la misma corrida, sin conservar audio de cola y con espera de 64 ms dio 556 ms y con espera de 32 ms 558 ms: la versión actual está entre 14 y 16 ms por encima en la mediana, dentro del ruido de una Mac ocupada pero no gratis. Se conserva ese audio de cola porque el reconocimiento perdía la última sílaba en finales suaves («policial» salía «policia»); `OPENTRADUCTOR_ASR_TAIL_KEEP_MS=0` lo apaga. Medición anterior con otro código (commit `4806494`, espera de 64 ms): 562 ms con traducción por cláusulas y 599 ms sin ella. WER del reconocimiento: 0,047. Términos bien traducidos en 7 charlas cortas escritas por el proyecto: 82,3 % sin contexto y 93,8 % con unos 200 tokens de charla previa (96 términos, medidos en T-80 sobre `4806494`; el audio de cola solo cambia lo que recibe el reconocimiento). Detalle en [docs/latency-final.md](docs/latency-final.md). Son muestras chicas, no resultados certificados. No se probó con micrófono físico de punta a punta, sesiones de más de 30 minutos ni Meet o Discord con personas reales.

### Probado en charlas reales

Con frases sueltas no se ve el caso difícil: en una charla el orador casi no para y la voz en inglés puede atrasarse y quedarse atrasada. Pasamos recortes de 11 minutos (660 s) de tres charlas reales en español por la app empaquetada, por el mismo camino que un micrófono (entrada BlackHole, salida a los parlantes de la Mac), y medimos cada frase desde el fin de habla hasta que empieza su voz en inglés. El audio no está en este repositorio. Fuentes, las tres con licencia CC BY 3.0:

1. T3chFest 2018, Daniel Jiménez, «Cómo se desarrolla software en el siglo XXI» ([video](https://www.youtube.com/watch?v=JoBv8sCuM3I))
2. T3chFest 2018, David Bonilla, «¿Cobras poco? Carreras profesionales en software» ([video](https://www.youtube.com/watch?v=d2_CUOeCgVE))
3. PyAr 2019, Hernán Wilkinson, «El desarrollo de software como debería ser» ([video](https://www.youtube.com/watch?v=lV9ozQbYEnc))

**Qué cambió en la etapa 3.** Cuatro cambios, encendidos por defecto, cada uno con su interruptor:

- **No se descarta nada.** La cola de entrada, el pendiente del reconocimiento y la cola de frases no tienen tope, y el Player ya no salta frases viejas ni corta la que está sonando. El atraso se absorbe acelerando la voz en inglés: 1,0x con hasta 1 s de cola y hasta 1,5x con 4 s o más. `OPENTRADUCTOR_NO_DROP=0` vuelve al comportamiento anterior de saltar y cortar (`tts_catchup`, descrito en la primera medición más abajo).
- **Voz simultánea.** Cada trozo de la frase se dice apenas hay 6 palabras confirmadas, no solo en comas y nexos, así la voz habla mientras el orador sigue hablando. `OPENTRADUCTOR_SIMUL_VOICE=0` la apaga; `OPENTRADUCTOR_SIMUL_VOICE_K` cambia el 6.
- **Voz al ritmo del orador.** La velocidad de cada frase se ajusta para que dure más o menos lo que duró el habla, entre 1,0x y 1,3x; se usa la mayor entre esta y la velocidad por atraso. `OPENTRADUCTOR_PACE_VOICE=0` la apaga; `OPENTRADUCTOR_PACE_MAX` cambia el 1,3x.
- **Tope al pendiente del reconocimiento.** Si al empezar una frase el audio que espera reconocimiento pasa de 20 s, se saltan frases enteras sin reconocerlas (ni subtítulo ni voz) hasta que baja a 3 s. Es la única pérdida que queda. `OPENTRADUCTOR_ASR_SKIP_OLD_S=0` lo apaga.

**Atraso de la voz (fin de habla hasta el audio en inglés), con los valores por defecto de la etapa 3.** Charla 1 (11 min, 660 s), mismo método que antes, dos pasadas de la app empaquetada:

| Charla 1 | Atraso p50 | Atraso p95 | Peor | Deriva | Audio perdido |
|---|---|---|---|---|---|
| Antes de la etapa 3 (dos corridas) | 2,9 y 3,0 s | 5,6 y 6,1 s | sin dato | ninguna (-0,02 y 0,00 s/min) | unos 2,8 s de voz saltada |
| Etapa 3, pasada 1 (commit `e8d9865`, ritmo 1,3x puesto a mano; `a54a00a` lo deja por defecto) | 2,3 s | 4,6 s | 7,0 s | ninguna (-0,06 s/min) | 0 (sin saltos, sin descartes, sin desborde) |
| Etapa 3 por defecto, pasada 2 (commit `a54a00a`, todo encendido por defecto) | 2,2 s | 4,3 s | 7,7 s | ninguna (0,00 s/min) | 0 |

La mediana bajó unos 0,7 s y el p95 entre 1 y 1,8 s, sin saltar audio. El costo es una voz más rápida: su velocidad llegó a 1,5x en todas las corridas, y en la pasada 1 estuvo por encima de 1,3x cerca del 12 % del tiempo de voz (estimación reconstruida del registro del monitor, no una medida directa). Aislando el cambio de ritmo del orador en la app (mismo build, una pasada apagado y una encendido): mediana de 2,67 a 2,29 s, p95 de 5,06 a 4,56 s. La pasada con el ritmo apagado corrió con la Mac más ocupada (carga media hasta 9,7 contra 7,8), así que parte de esa mejora puede ser ruido.

**Audio perdido, con la Mac saturada.** Una sesión de 30 minutos (mezcla de las tres charlas) con 36 procesos usando la CPU del minuto 5 al 25, carga media hasta 126:

| Medida | Valor |
|---|---|
| Atraso p50 / p95 / peor (663 frases) | 2,6 / 5,1 / 8,4 s |
| Minutos 15 a 30 (con carga hasta el minuto 25) p50 / p95 / peor | 3,0 / 5,4 / 8,4 s |
| Deriva | +0,04 s/min |
| Pendiente del reconocimiento, peor | 0,32 s |
| Frases saltadas por el tope | 0 de 690 |
| Desborde de audio, audio descartado, errores de salida | 0 |

Una versión anterior sin el tope, con el mismo tipo de carga (carga media hasta 50, 20 minutos), se atrasó sin límite: el atraso llegó a 730 s y el pendiente del reconocimiento a 337 s. En esta corrida el pendiente nunca se acercó a 20 s, así que el tope no actuó ni una vez (0 eventos). Muestra que el problema no se reprodujo, no que el tope funcione: el tope solo tiene pruebas con un reconocedor lento simulado (el pendiente quedó en 22 s como máximo en 300 s de habla con reconocimiento al 70 % del tiempo real) y ninguna corrida de punta a punta en la app empaquetada. No sabemos por qué la app se afectó menos con la carga esta vez (usó 53 % de la CPU bajo carga contra 97 % en la prueba anterior).

El atraso sigue siendo de segundos, no el de menos de un segundo de la tabla de arriba, y no prometemos 1 segundo. La cifra de menos de un segundo vale para frases aisladas.

El atraso de la versión anterior, como referencia. Antes de la etapa 3 la voz en inglés hacía cola cuando el orador hablaba más rápido de lo que Kokoro podía decir. `tts_catchup` (`OPENTRADUCTOR_TTS_CATCHUP=0` lo apaga) aceleraba Kokoro hasta 1,30x cuando crecía la cola, recortaba a 80 ms el silencio que Kokoro agrega en cada trozo, saltaba frases viejas enteras cuando la cola pasaba de 8 s y cortaba las cláusulas todavía no dichas de la frase que sonaba si llegaba una nueva con más de 5 s de cola. Las frases con cifras o nombres propios no se saltaban ni se cortaban. Esa versión, en tres charlas:

| Charla | Atraso p50 | Atraso p95 | Deriva | Voz saltada |
|---|---|---|---|---|
| 1 (dos corridas) | 2,9 y 3,0 s | 5,6 y 6,1 s | ninguna (-0,02 y 0,00 s/min) | unos 2,8 s de 660 s |
| 2 (una corrida, commit `ff71322`) | 4,0 s | 7,8 s | ninguna (+0,03 s/min) | 13,8 s (2,1 %) |
| 3 (una corrida, commit `ff71322`) | 3,8 s | 7,1 s | ninguna (-0,10 s/min) | 11,7 s (1,8 %) |

Las charlas 2 y 3 no se volvieron a correr con los valores por defecto de la etapa 3.

Límites de las cifras de la etapa 3:
- Una pasada por variante (dos para la charla 1 con los valores finales, una de ellas antes de que existiera el tope). Dos corridas del mismo código pueden diferir en varias décimas de segundo en el p95.
- Una Mac (M5 Pro). Las charlas son grabaciones reproducidas por BlackHole, no una reunión con personas.
- Nadie juzgó escuchando si la voz más rápida suena natural. Solo medimos el atraso.
- La prueba con la Mac saturada fue una sola corrida de 30 minutos, y el tope nunca se activó en ella.
- Con el tope encendido, un pendiente muy grande salta frases enteras; con `OPENTRADUCTOR_ASR_SKIP_OLD_S=0` no se salta nada pero el atraso puede crecer sin límite.

**Términos técnicos en una charla real.** En la charla 1 contamos 47 términos técnicos («tests de aceptación», «management 3.0», «pipeline», etc.) con expresiones regulares escritas por el proyecto, en dos corridas de la app antes y después del cambio de vocabulario: 22,5 de 47 (48 %) antes y 28 de 47 (60 %) después. El cambio tiene dos partes: el reconocedor recibe un glosario (`runtime/data/asr_keyterms.txt`) como pistas, y sus resultados parciales se corrigen contra ese glosario por coincidencia aproximada. Las dos están encendidas por defecto (`OPENTRADUCTOR_ASR_KEYTERMS=0` y `OPENTRADUCTOR_ASR_GLOSSARY_FIX=0` las apagan).

Límites de esta cifra:
- Es una sola charla, pasada una o dos veces por configuración. Dos corridas del mismo código difieren en 3 a 4 términos: leé la mejora como 60 % más o menos 4 puntos.
- Las expresiones regulares son nuestras y nadie más las revisó.
- El glosario lo armó el proyecto con vocabulario de este tipo de charla (ágil, testing, DevOps) y la charla 1 trata ese tema. En otros temas esperá menos. En una nueva puntuación offline de la charla 2, de otro tema, la cuenta no cambió (25 de 35 términos).
- En las corridas en tiempo real la mejora vino de la corrección por glosario. Con las pistas al reconocedor apagadas (`OPENTRADUCTOR_ASR_KEYTERMS=0`) salieron 29 de 47 en una corrida, igual que con ellas encendidas.
- Con los valores por defecto de la etapa 3, la charla 1 dio 29 y 32 de 47 términos en las dos pasadas, dentro del mismo ruido de 3 a 4 términos. La etapa 3 no cambia el texto traducido.
- Siglas y palabras en inglés dichas dentro del español (ATDD, waterfall, lean, DevOps) siguen mal traducidas.
- La Mac estaba ocupada durante las corridas (carga media entre 4 y 12).


## Cómo contribuir

Abrí un issue o un pull request. Las áreas más útiles son la latencia, la calidad con español rioplatense (un set de prueba más grande, glosario del usuario) y la portabilidad (instalación en una Mac limpia). Corré `uv run --extra runtime python -m pytest tests -q` y contá qué Mac y qué macOS usaste. El audio de evaluación (`data/audio/`) no está en el repositorio: en un clon nuevo los tests que lo leen se saltean con el motivo «data/audio ausente». Para bajarlo, mirá la sección «Run the tests» de [README.md](README.md).

## Licencia

Código bajo [MIT](LICENSE). Los modelos y los datos mantienen sus licencias, listadas en [NOTICE.md](NOTICE.md).

La voz inglesa de Kokoro usa `phonemizer-fork` y espeak-ng (GPL-3.0). El repo solo los declara como dependencias. Si redistribuís un binario que los incluya, ese binario debe cumplir la GPL-3.0. Detalle en [NOTICE.md](NOTICE.md).
