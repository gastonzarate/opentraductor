# opentraductor (resumen en español)

Traductor de voz **local** de español argentino a inglés de EE. UU. para macOS con Apple Silicon. Hablás en español y la Mac lo dice en inglés un instante después. Todo corre en tu equipo: reconocimiento de voz, traducción y síntesis.

> **Estado: experimental.** Hoy la demora entre el fin de tu frase y el primer audio en inglés es de poco menos de un segundo. No hay promesa de 300 ms. Documentación completa en inglés: [README.md](README.md).

## Qué hace

Micrófono → VAD (Silero) → reconocimiento (Parakeet o Moonshine) → traducción (Qwen3 o Marian) → voz (Kokoro) → micrófono virtual para Meet o Discord.

- Sin red después de bajar los modelos. Sin telemetría ni cuentas.
- No guarda audio, solo textos en un historial local que podés exportar o borrar.
- Interfaz web local en `127.0.0.1`.

## Requisitos

Mac con Apple Silicon (probado en M5 Pro, macOS 26), Python 3.12 y [uv](https://docs.astral.sh/uv/). Para reuniones, BlackHole 2ch (GPL-3.0, no incluido: lo instalás vos).

## Instalación

```bash
git clone https://github.com/<owner>/opentraductor.git
cd opentraductor
uv sync --extra runtime
uv run --extra runtime python -m runtime.setup    # descarga modelos (único paso con red)
uv run --extra runtime python -m runtime check
uv run --extra runtime python -m runtime run      # abre http://127.0.0.1:8877/
```

El perfil más rápido (Moonshine + Qwen3) y el micrófono virtual están explicados en el README en inglés.

## Resultados medidos

Mediana desde el fin de habla hasta el primer audio: unos 930 ms (p95 cerca de 1 s), 21 corridas con 3 audios, en una Mac con otras tareas abiertas. WER del reconocimiento: 0,047. Términos bien traducidos: 69 % sin contexto y 77 % con unos 500 tokens de charla previa. Son muestras chicas, no resultados certificados. No se probó con micrófono físico de punta a punta, sesiones largas ni Meet o Discord con personas reales.

## Cómo contribuir

Abrí un issue o un pull request. Las áreas más útiles son la latencia, la calidad con español rioplatense (un set de prueba más grande, glosario del usuario) y la portabilidad (instalación en una Mac limpia). Corré `uv run --extra runtime python -m pytest tests -q` y contá qué Mac y qué macOS usaste. El audio de evaluación (`data/audio/`) no está en el repositorio: en un clon nuevo los tests que lo leen se saltean con el motivo «data/audio ausente». Para bajarlo, mirá la sección «Run the tests» de [README.md](README.md).

## Licencia

Código bajo [MIT](LICENSE). Los modelos y los datos mantienen sus licencias, listadas en [NOTICE.md](NOTICE.md).

La voz inglesa de Kokoro usa `phonemizer-fork` y espeak-ng (GPL-3.0). El repo solo los declara como dependencias. Si redistribuís un binario que los incluya, ese binario debe cumplir la GPL-3.0. Detalle en [NOTICE.md](NOTICE.md).
