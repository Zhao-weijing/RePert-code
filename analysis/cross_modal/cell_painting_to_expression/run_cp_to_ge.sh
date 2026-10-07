#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/path/to/home/.conda/envs/perturbnet310/bin/python3.10"
SCRIPT="/path/to/data/AIDD/MVCPert_5_27/runs/multimodal_expansion_20260830/02_CP_to_GE/cp_to_ge_residual.py"
ROOT="/path/to/data/AIDD/MVCPert_5_27/runs/multimodal_expansion_20260830/02_CP_to_GE/formal"

test ! -e "$ROOT/seed_3407"
test ! -e "$ROOT/seed_42"
test ! -e "$ROOT/seed_2025"

for seed in 3407 42 2025; do
  "$PYTHON_BIN" "$SCRIPT" --seed "$seed" --outdir "$ROOT/seed_$seed" --null-rounds 32 --bootstrap-rounds 10000
done

"$PYTHON_BIN" "$SCRIPT" --aggregate-root "$ROOT" --bootstrap-rounds 10000
