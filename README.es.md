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

Mediana desde el fin de habla hasta el primer audio con el código de este repositorio: 572 ms, 64 corridas (8 audios x 8 rondas, A/B intercalado), en una Mac con otras tareas abiertas (carga media 3,6, hasta 5,9). Con la Mac ocupada hay picos de cerca de 1 segundo (p95 1039 ms en esa corrida). En la misma corrida, sin conservar audio de cola y con espera de 64 ms dio 556 ms y con espera de 32 ms 558 ms: la versión actual está entre 14 y 16 ms por encima en la mediana, dentro del ruido de una Mac ocupada pero no gratis. Se conserva ese audio de cola porque el reconocimiento perdía la última sílaba en finales suaves («policial» salía «policia»); `OPENTRADUCTOR_ASR_TAIL_KEEP_MS=0` lo apaga. Medición anterior con otro código (commit `4806494`, espera de 64 ms): 562 ms con traducción por cláusulas y 599 ms sin ella. WER del reconocimiento: 0,047. Términos bien traducidos en 7 charlas cortas escritas por el proyecto: 82,3 % sin contexto y 93,8 % con unos 200 tokens de charla previa (96 términos, medidos en T-80 sobre `4806494`; el audio de cola solo cambia lo que recibe el reconocimiento). Detalle en [docs/latency-final.md](docs/latency-final.md). Son muestras chicas, no resultados certificados. No se probó con micrófono físico de punta a punta, sesiones de más de 11 minutos ni Meet o Discord con personas reales.

### Probado en charlas reales

Con frases sueltas no se ve el caso difícil: en una charla el orador casi no para y la voz en inglés puede atrasarse y quedarse atrasada. Pasamos recortes de 11 minutos (660 s) de tres charlas reales en español por la app empaquetada, por el mismo camino que un micrófono (entrada BlackHole, salida a los parlantes de la Mac), y medimos cada frase desde el fin de habla hasta que empieza su voz en inglés. El audio no está en este repositorio. Fuentes, las tres con licencia CC BY 3.0:

1. T3chFest 2018, Daniel Jiménez, «Cómo se desarrolla software en el siglo XXI» ([video](https://www.youtube.com/watch?v=JoBv8sCuM3I))
2. T3chFest 2018, David Bonilla, «¿Cobras poco? Carreras profesionales en software» ([video](https://www.youtube.com/watch?v=d2_CUOeCgVE))
3. PyAr 2019, Hernán Wilkinson, «El desarrollo de software como debería ser» ([video](https://www.youtube.com/watch?v=lV9ozQbYEnc))

**Atraso de la voz (fin de habla hasta el audio en inglés).** La voz en inglés hace cola cuando el orador habla más rápido de lo que Kokoro puede decir. `tts_catchup` (encendido por defecto; `OPENTRADUCTOR_TTS_CATCHUP=0` lo apaga) acelera Kokoro hasta 1,30x cuando crece la cola, recorta a 80 ms el silencio que Kokoro agrega en cada trozo, salta frases viejas enteras cuando la cola pasa de 8 s y, si llega una frase nueva con más de 5 s de cola, corta las cláusulas todavía no dichas de la que está sonando (nunca más de 3 s ni más de la mitad de la frase). Las frases con cifras o nombres propios no se saltan ni se cortan. Una frase saltada igual aparece en los subtítulos.

| Charla | Atraso p50 | Atraso p95 | Deriva | Voz saltada |
|---|---|---|---|---|
| 1 (dos corridas, runtime actual) | 2,9 y 3,0 s | 5,6 y 6,1 s | ninguna (-0,02 y 0,00 s/min) | unos 2,8 s de 660 s |
| 2 (una corrida, commit `ff71322`) | 4,0 s | 7,8 s | ninguna (+0,03 s/min) | 13,8 s (2,1 %) |
| 3 (una corrida, commit `ff71322`) | 3,8 s | 7,1 s | ninguna (-0,10 s/min) | 11,7 s (1,8 %) |

El atraso se mantiene estable en los 11 minutos (sin deriva) y no se perdió audio (sin desborde ni errores del dispositivo de salida). No es corto: unos 3 a 4 s en la mediana y 6 a 8 s en las peores frases. Nuestra meta era un p95 de 6 s o menos: la charla 1 queda en el borde y las charlas 2 y 3 no la cumplen. Es un atraso de segundos, no el de menos de un segundo de la tabla de arriba, y no prometemos uno menor. La cifra de menos de un segundo vale para frases aisladas, cuando la voz no va atrasada.

**Términos técnicos en una charla real.** En la charla 1 contamos 47 términos técnicos («tests de aceptación», «management 3.0», «pipeline», etc.) con expresiones regulares escritas por el proyecto, en dos corridas de la app antes y después del cambio de vocabulario: 22,5 de 47 (48 %) antes y 28 de 47 (60 %) después. El cambio tiene dos partes: el reconocedor recibe un glosario (`runtime/data/asr_keyterms.txt`) como pistas, y sus resultados parciales se corrigen contra ese glosario por coincidencia aproximada. Las dos están encendidas por defecto (`OPENTRADUCTOR_ASR_KEYTERMS=0` y `OPENTRADUCTOR_ASR_GLOSSARY_FIX=0` las apagan).

Límites de esta cifra:
- Es una sola charla, pasada una o dos veces por configuración. Dos corridas del mismo código difieren en 3 a 4 términos: leé la mejora como 60 % más o menos 4 puntos.
- Las expresiones regulares son nuestras y nadie más las revisó.
- El glosario lo armó el proyecto con vocabulario de este tipo de charla (ágil, testing, DevOps) y la charla 1 trata ese tema. En otros temas esperá menos. En una nueva puntuación offline de la charla 2, de otro tema, la cuenta no cambió (25 de 35 términos).
- En las corridas en tiempo real la mejora vino de la corrección por glosario. Con las pistas al reconocedor apagadas (`OPENTRADUCTOR_ASR_KEYTERMS=0`) salieron 29 de 47 en una corrida, igual que con ellas encendidas.
- Siglas y palabras en inglés dichas dentro del español (ATDD, waterfall, lean, DevOps) siguen mal traducidas.
- La Mac estaba ocupada durante las corridas (carga media entre 4 y 12).


## Cómo contribuir

Abrí un issue o un pull request. Las áreas más útiles son la latencia, la calidad con español rioplatense (un set de prueba más grande, glosario del usuario) y la portabilidad (instalación en una Mac limpia). Corré `uv run --extra runtime python -m pytest tests -q` y contá qué Mac y qué macOS usaste. El audio de evaluación (`data/audio/`) no está en el repositorio: en un clon nuevo los tests que lo leen se saltean con el motivo «data/audio ausente». Para bajarlo, mirá la sección «Run the tests» de [README.md](README.md).

## Licencia

Código bajo [MIT](LICENSE). Los modelos y los datos mantienen sus licencias, listadas en [NOTICE.md](NOTICE.md).

La voz inglesa de Kokoro usa `phonemizer-fork` y espeak-ng (GPL-3.0). El repo solo los declara como dependencias. Si redistribuís un binario que los incluya, ese binario debe cumplir la GPL-3.0. Detalle en [NOTICE.md](NOTICE.md).
