# Corpus de audio humano (benchmark)

Owner: worker de corpus (Hub, equipo 14750190). Contrato: `benchmark/contracts.py`, `bench-contract-coord-v1`.

## Muestra 1: FLEURS es_419 / en_us (validation)

- Fuente: `google/fleurs` en Hugging Face, <https://huggingface.co/datasets/google/fleurs>.
- Licencia: CC-BY-4.0, <https://creativecommons.org/licenses/by/4.0/>. Atribución en el manifest.
- Cita: Conneau et al. (2022), FLEURS.
- Script: `scripts/datasets/fetch_fleurs_sample.py`.
  Comando: `uv run --with pyarrow python scripts/datasets/fetch_fleurs_sample.py --n 10 --seed 20260514`.
  `pyarrow` se usa solo en ese comando efímero; no está en `pyproject.toml`.
- Revisión del dataset (commit HF): `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, modificado 2026-05-15.
- Parquet (cache en `~/.cache/opentraductor/fleurs`):
  - es_419 validation, sha256 `4318ab0fdf28bc413a209b77afc19b1d0b466e9166e5852b217de5fe94225f1c`
  - en_us validation, sha256 `7c3eeb11a9597bd52cdc1b0d637e85389fe094cfd8763913e7bf4fdf7a853959`
- Selección: semilla 20260514 sobre los ids paralelos ordenados (150). Ids elegidos: 1516, 1549, 1553, 1556, 1565, 1584, 1593, 1612, 1624, 1652.
- Duplicados: hay 408 filas es_419 para 150 frases. Cada id tiene 1 a 3 grabaciones con audio distinto (probablemente voces distintas). La muestra usa solo la primera aparición de cada id en el parquet; las demás no se usan (258 en es_419, 244 en en_us).
- Conversión: los WAV originales son float. Se convierten a PCM16 mono 16 kHz con `sf.read` y `benchmark.audio.write_wav_pcm16_mono_16k`. El `sha256` del manifest es del archivo convertido. El hash del original queda en `source_audio_sha256`.
- Salida: 10 WAV en `data/audio/fleurs/es_419/validation/`, 103.5 s en total. Manifest: `data/manifests/fleurs_es_sample.jsonl`. Metadata con hashes, revisión y semilla: `data/manifests/fleurs_es_sample.metadata.json`.
- Validación: `contracts.load_manifest(verify_sha256=True)` y `benchmark.runner.items_from_manifest` pasan con 10 items.
- Orden del pipeline: `fetch_fleurs_sample.py` y después `estimate_boundaries.py --manifest data/manifests/fleurs_es_sample.jsonl --audio-root data/audio --metadata data/manifests/fleurs_es_sample.metadata.json`. Si se regenera el manifest, hay que volver a correr el estimador.
- Split: `development` (no se tocó `test`). Esta muestra es piloto y no es un test independiente.

### Campos y limitaciones

- `dialect: unknown`. El dato es es_419, pero eso no demuestra que excluya el acento argentino. No se etiqueta como es-AR.
- `speaker_id: null`. La versión de HF no trae identificador de hablante, así que la agrupación es conservadora por fuente (`unknown-source:fleurs`). Por eso esta muestra es solo development.
- `transcript_es`: transcripción humana (`raw_transcription`).
- `reference_en`: texto en_us con el mismo id FLEURS. Es un texto paralelo humano. No hay garantía de que sea traducción directa del audio es_419 (ver `reference_note` en cada item).
- `boundary_method: estimated` en 8 items, con `boundary_uncertainty_s: 0.3`. Los 0.3 s son una tolerancia SUPUESTA, no una cota validada ni un intervalo de confianza. Sin ground truth no hay forma de validarla.
- `boundary_method: unknown` en 1553 y 1565 (`late_onset`, inicio > 1.0 s). Un tramo previo de baja energía puede ser habla suave. Ver la regla en el estimador.
- Método: `scripts/datasets/estimate_boundaries.py` calcula la energía por bloque de 10 ms, con umbral = piso de ruido (p10) + 10 dB. No usa el candidato ni modelos. No son límites humanos ni alineación revisada. Parámetros y plausibilidad por item en el metadata JSON (`boundaries`).
- Plausibilidad: 10.5 a 18.3 caracteres por segundo de habla. Sin ground truth, no hay precisión medida.
- Riesgo: el estimador trata como ruido el tramo de -40 a -47 dBFS antes de la voz. En los items marcados como `late_onset` puede haber habla suave, así que el error puede superar 0.3 s. Otros items con inicio entre 0.3 y 0.9 s no se revisaron a mano.
- Cobertura: lectura de frases de Wikipedia/noticias. No hay habla conversacional, ruido de fondo, ni cambios de idioma en esta muestra.
- Faltantes: acento argentino, referencias de traductor independiente, speakers identificados, test congelado.

## Pendiente

- Ampliar muestra y declarar cobertura real (números, nombres, negaciones).
- Candidatos: CoVoST 2 es→en (si el acceso es público), Common Voice (si hay descarga pública con licencia y cuota de español argentino).
- Test congelado, con split por hablante cuando la fuente lo permita.

## Búsqueda de muestra held-out con hablantes identificados (es-AR)

Tiempo acotado. Resultado: sin fuente utilizable hoy.

- Mozilla Common Voice (es): el repo HF `mozilla-foundation/common_voice_17_0` solo tiene README. Desde octubre de 2025 Common Voice se distribuye solo por Mozilla Data Collective (<https://datacollective.mozillafoundation.org>), que exige cuenta. No se usó cuenta ni aceptación de condiciones: eso requiere coordinador.
- CoVoST 2 es→en (`facebook/covost2`, licencia de metadatos CC-BY-NC-4.0, uso no comercial): el TSV `covost_v2.es_en.tsv.tar.gz` está accesible (112 127 filas, columnas `path`, `translation`, `split`). No trae speaker ID ni client_id. El audio apunta a clips de Common Voice, que quedan detrás de Mozilla Data Collective. No hay audio descargable sin cuenta.
- FLEURS en HF no trae speaker ID (ver arriba).
- Conclusión: no hay hablantes identificados ni audio es-AR redistribuible verificable. No se inventó cobertura. El split por hablante queda como `unknown-source` para todo el corpus actual.

## Set de texto argentino (desarrollo, sin audio)

- Archivo: `data/manifests/text_meetings_ar_dev.jsonl`, 54 frases en español rioplatense sobre reuniones ficticias de software y tecnología (dominio confirmado por el usuario, ver `docs/requirements.md`).
- Incluye pares mínimos (`minimal_pair`) donde un cambio altera el significado: "hay que" / "no hace falta", "antes" / "después", "0,3 segundos" / "3 segundos", "1.2.3" / "1.2.4", "1.250" / "125".
- Cubre voseo, negación, números (coma decimal, separador de miles con punto), nombres propios, términos técnicos en inglés, ambigüedad, PR/merge/deploy/rollback, API, incidentes, versiones semánticas, latencias y unidades. Cada item lista sus fenómenos en `phenomena`.
- `transcript_es_provenance` y `reference_en_provenance` son `generated`. Las referencias en inglés las escribió el worker de corpus, no son traducciones humanas. No se usa TTS como audio humano.
- Split: `development`. No es test final ni contiene datos de conversaciones reales.

## Muestra 2: OpenSLR 61, español argentino (es-AR), CC-BY-SA-4.0

- Fuente primaria: <https://openslr.org/61/> (Guevara-Rukoz et al., 2020, LREC). Google, voluntarios de Buenos Aires. 5 739 grabaciones de índice, 44 hablantes en total (31 mujeres, 13 hombres, según los prefijos y los índices).
- Licencia: CC-BY-SA-4.0. Copyleft: los derivados (manifest, extractos) quedan bajo CC-BY-SA. Por eso van en `data/manifests/openslr61_es_ar_sample.jsonl` y `data/audio/openslr61/`, separados de la muestra FLEURS (CC-BY). Revisar antes de mezclar con cualquier otra licencia.
- Índices (`line_index_female.tsv`, `line_index_male.tsv`) y zip por género leídos por HTTP Range. No se descargó el zip completo: el de mujeres pesa 1.2 GB y el de hombres 551 MB.
- Script: `scripts/datasets/fetch_openslr61_sample.py --seed 20260514 --per-speaker 3`. Usa numpy para remuestrear de 48 kHz a 16 kHz (FIR sinc, corte 7 kHz). Sin dependencias nuevas.
- Selección: semilla 20260514. 4 hablantes para development (2 mujeres, 2 hombres) y 4 para test (2 mujeres, 2 hombres), disjuntos. 3 clips por hablante. Total 24 clips, 115.5 s.
- Speaker ID: segundo campo del FileID, con género como prefijo (`female_02121`). Los IDs no se usan sin género.
- Dialecto: `es-AR`, según el metadato de la fuente. No se verificó con un lingüista.
- Transcripciones: `transcript_es_provenance: human`, según el readme de la fuente ("manually quality checked"). Pueden tener errores.
- Referencias en inglés: no existen en esta fuente. `reference_en: null`, `reference_en_provenance: unknown`. Es válido por el contrato y no se mide traducción con esta muestra.
- Límites: `estimated` en 21 items; `unknown` en 3 (`implausible_rate`: velocidad fuera de 6 a 20 caracteres por segundo). Ver la regla en el estimador.
- Validación: `load_manifest(verify_sha256=True)` e `items_from_manifest` pasan con 24 items.
- Consentimiento: la fuente no publica consentimiento individual en el índice. Se anota en `consent_note`.

## Revisión de la regla de límites

`scripts/datasets/estimate_boundaries.py` ahora marca como `unknown` un item con inicio mayor que 1.0 s (`late_onset`) o con velocidad fuera de 6 a 20 caracteres por segundo (`implausible_rate`). Afecta a FLEURS (1553 y 1565 por inicio tardío) y a OpenSLR (3 items por velocidad). Los 0.3 s siguen siendo una tolerancia supuesta, sin validar.
