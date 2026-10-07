#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/path/to/home/.conda/envs/perturbnet310/bin/python3.10"
SCRIPT="/path/to/data/AIDD/MVCPert_5_27/runs/multimodal_expansion_20260830/04_sequential_evidence/sequential_evidence.py"
ROOT="/path/to/data/AIDD/MVCPert_5_27/runs/multimodal_expansion_20260830/04_sequential_evidence/formal_retry"

test ! -e "$ROOT/seed3407"
test ! -e "$ROOT/seed42"
test ! -e "$ROOT/seed2025"

for seed in 3407 42 2025; do
  "$PYTHON_BIN" "$SCRIPT" --seed "$seed" --outdir "$ROOT/seed$seed" --null-rounds 32 --bootstrap-rounds 10000
done

"$PYTHON_BIN" "$SCRIPT" --aggregate-root "$ROOT" --bootstrap-rounds 10000
