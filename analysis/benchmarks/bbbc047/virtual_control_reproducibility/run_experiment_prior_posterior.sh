#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${MVCPERT_PYTHON:-/path/to/home/.conda/envs/perturbnet310/bin/python3.10}"
OUTPUT_ROOT="${MVCPERT_PRIOR_POSTERIOR_OUTPUT_ROOT:-/path/to/data/AIDD/MVCPert_5_27/runs/repro_effect_prior_posterior_20260829}"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
for seed in 3407 42 2025; do "$PYTHON_BIN" repro_effect_prior_posterior.py --seed "$seed" --outdir "$OUTPUT_ROOT/seed${seed}"; done
"$PYTHON_BIN" repro_effect_prior_posterior.py --aggregate-root "$OUTPUT_ROOT"
