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

Mediana desde el fin de habla hasta el primer audio con el código de este repositorio: 572 ms, 64 corridas (8 audios x 8 rondas, A/B intercalado), en una Mac con otras tareas abiertas (carga media 3,6, hasta 5,9). Con la Mac ocupada hay picos de cerca de 1 segundo (p95 1039 ms en esa corrida). En la misma corrida, sin conservar audio de cola y con espera de 64 ms dio 556 ms y con espera de 32 ms 558 ms: la versión actual está entre 14 y 16 ms por encima en la mediana, dentro del ruido de una Mac ocupada pero no gratis. Se conserva ese audio de cola porque el reconocimiento perdía la última sílaba en finales suaves («policial» salía «policia»); `OPENTRADUCTOR_ASR_TAIL_KEEP_MS=0` lo apaga. Medición anterior con otro código (commit `4806494`, espera de 64 ms): 562 ms con traducción por cláusulas y 599 ms sin ella. WER del reconocimiento: 0,047. Términos bien traducidos en 7 charlas cortas escritas por el proyecto: 82,3 % sin contexto y 93,8 % con unos 200 tokens de charla previa (96 términos, medidos en T-80 sobre `4806494`; el audio de cola solo cambia lo que recibe el reconocimiento). Detalle en [docs/latency-final.md](docs/latency-final.md). Son muestras chicas, no resultados certificados. No se probó con micrófono físico de punta a punta, sesiones largas ni Meet o Discord con personas reales.

## Cómo contribuir

Abrí un issue o un pull request. Las áreas más útiles son la latencia, la calidad con español rioplatense (un set de prueba más grande, glosario del usuario) y la portabilidad (instalación en una Mac limpia). Corré `uv run --extra runtime python -m pytest tests -q` y contá qué Mac y qué macOS usaste. El audio de evaluación (`data/audio/`) no está en el repositorio: en un clon nuevo los tests que lo leen se saltean con el motivo «data/audio ausente». Para bajarlo, mirá la sección «Run the tests» de [README.md](README.md).

## Licencia

Código bajo [MIT](LICENSE). Los modelos y los datos mantienen sus licencias, listadas en [NOTICE.md](NOTICE.md).

La voz inglesa de Kokoro usa `phonemizer-fork` y espeak-ng (GPL-3.0). El repo solo los declara como dependencias. Si redistribuís un binario que los incluya, ese binario debe cumplir la GPL-3.0. Detalle en [NOTICE.md](NOTICE.md).
