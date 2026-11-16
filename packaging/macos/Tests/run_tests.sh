#!/bin/zsh
# Pruebas de la lógica sin UI de la app (T-55: Transcript + EngineEvent.parse).
# Solo hace falta Command Line Tools (sin Xcode ni XCTest).
set -euo pipefail
HERE="${0:A:h}"
SRC="$HERE/../Sources/OpenTraductor"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
swiftc -target arm64-apple-macos15.0 -o "$OUT/transcript_tests" \
  "$SRC/Transcript.swift" "$SRC/EngineClient.swift" "$HERE/main.swift"
# Fixture con eventos del runtime real (T-55). Regenerar: .venv/bin/python -m tests.integration.t55_dump_events
"$OUT/transcript_tests" "$HERE/../../../tests/fixtures/t55_runtime_events.json"
