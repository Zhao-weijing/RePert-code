#!/usr/bin/env bash
set -euo pipefail

ROOT=/path/to/data/AIDD/MVCPert_5_27/runs/one_r_ge_residual_evidence_20260829
SCRIPT_DIR=/path/to/data/AIDD/MVCPert_5_27/analysis/benchmarks/bbbc047/single_repeat_expression_residual

for seed in 3407 42 2025; do
  /path/to/home/.conda/envs/perturbnet310/bin/python3.10 \
    "$SCRIPT_DIR/evaluate_1r_ge_residual.py" \
    --seed "$seed" --device cuda --outdir "$ROOT/seed${seed}"
done

/path/to/home/.conda/envs/perturbnet310/bin/python3.10 \
  "$SCRIPT_DIR/evaluate_1r_ge_residual.py" --aggregate-root "$ROOT"
