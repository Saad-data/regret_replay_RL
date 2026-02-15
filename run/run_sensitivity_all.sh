#!/bin/bash
# 60 sensitivity runs. cartpole only.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"

CONFIGS="per_alpha_0.4 per_alpha_0.6 per_alpha_0.8 \
        per_beta_0.3 per_beta_0.4 per_beta_0.6 \
        temp_0.5 temp_1.0 temp_2.0 \
        lfiw_temp_5.0 lfiw_temp_10.0 lfiw_temp_15.0"

ENV="CartPole-v1"
SEEDS="0 1 2 3 4"
STEPS=200000

echo "========================================================================"
echo "SENSITIVITY ANALYSIS: 60 runs"
echo "========================================================================"
echo "Start: $(date)"

total=0

for config in $CONFIGS; do
    echo "[$config]"

    for seed in $SEEDS; do
        total=$((total + 1))
        echo "  [$total/60] seed $seed - $(date +'%H:%M:%S')"

        python run/run_sensitivity.py \
            --config "$config" \
            --env "$ENV" \
            --seed "$seed" \
            --steps "$STEPS"
    done
done

echo "Sensitivity complete: $(date)"
