#!/bin/bash
# 100 runs: 5 ablations x 10 seeds x 2 envs. run from project root

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"

CONFIGS="lfiw_only tce_only htd_lfiw htd_tce lfiw_tce"
ENVS="CartPole-v1 MountainCar-v0"
SEEDS="0 1 2 3 4 5 6 7 8 9"
STEPS=200000

echo "========================================================================"
echo "COMPLETE ABLATION MATRIX: 100 runs"
echo "========================================================================"
echo "Start: $(date)"

total=0

for config in $CONFIGS; do
    for env in $ENVS; do
        echo "[$config on $env]"

        for seed in $SEEDS; do
            total=$((total + 1))
            echo "  [$total/100] seed $seed - $(date +'%H:%M:%S')"

            python run/run_ablation.py \
                --config $config \
                --env $env \
                --seed $seed \
                --steps $STEPS
        done
    done
done

echo "Ablations complete: $(date)"
