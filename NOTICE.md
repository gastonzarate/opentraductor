# Third-party notices

opentraductor's own source code is licensed under the MIT License (see `LICENSE`).
This file lists third-party material that is bundled in this repository, downloaded by its scripts, or required at run time.

## Bundled in this repository

### Silero VAD (`runtime/data/silero_vad.onnx`)

MIT License. Copyright (c) 2020-present Silero Team. <https://github.com/snakers4/silero-vad>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Evaluation manifests (`data/manifests/`)

- `fleurs_es_sample.*`: derived from FLEURS (Google), CC-BY 4.0, <https://huggingface.co/datasets/google/fleurs>. Cite: Conneau et al. (2022), "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech".
- `openslr61_es_ar_sample.*`: derived from OpenSLR 61, High-quality TTS data for Argentinian Spanish (Guevara-Rukoz et al., 2020, LREC), <https://openslr.org/61/>. **CC-BY-SA 4.0**: these files and any extract of them stay under CC-BY-SA 4.0, not MIT.
- `text_meetings_ar_dev.jsonl`: written for this project, MIT.

The audio files are not part of the repository. Scripts in `scripts/datasets/` download them from the original sources.

## Downloaded at setup time (not redistributed here)

Model weights are fetched from their publishers by `python -m runtime.setup`, `scripts/setup_moonshine.py` or by the user. Each keeps its own license.

| Component | Model | License | Source |
|---|---|---|---|
| Speech recognition | NVIDIA Parakeet TDT 0.6B v3 (MLX conversion by mlx-community) | CC-BY 4.0. Attribution: NVIDIA. | <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3> |
| Speech recognition (optional) | Moonshine streaming Spanish (tiny, small) | MIT. Copyright (c) 2025 Useful Sensors, Inc. (dba Moonshine AI) | <https://github.com/moonshine-ai/moonshine> |
| Translation | Qwen3-4B-Instruct-2507 and Qwen3-1.7B (4-bit MLX conversions) | Apache-2.0. Copyright Alibaba Cloud / Qwen team. | <https://huggingface.co/Qwen> |
| Translation (baseline) | Helsinki-NLP opus-mt-es-en (Marian), converted to CTranslate2 int8 | Apache-2.0 | <https://huggingface.co/Helsinki-NLP/opus-mt-es-en> |
| Speech synthesis | Kokoro-82M (MLX conversion by mlx-community), en-US voices | Apache-2.0 | <https://huggingface.co/hexgrad/Kokoro-82M> |

## Python dependencies

Dependencies are installed by `uv` from PyPI, not copied here. Two of them are copyleft:

- `phonemizer-fork` (GPL-3.0-or-later) and the espeak-ng library shipped by `espeakng-loader` (GPL-3.0-or-later) are used for English grapheme-to-phoneme conversion in the Kokoro text-to-speech stage.
- `num2words` is LGPL.

Distributing this source code does not distribute those packages. **If you build and redistribute a binary (an app bundle, a disk image or an installer) that includes them, that binary must comply with the GPL-3.0** (provide the corresponding source and license texts). The MIT license of this repository does not change that.

## Optional external software

- **BlackHole** (virtual audio driver, GPL-3.0, <https://github.com/ExistentialAudio/BlackHole>) is not included, linked or installed by this project. You install it yourself if you want to route the translated voice to Meet or Discord.
