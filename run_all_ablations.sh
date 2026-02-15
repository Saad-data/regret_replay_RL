#!/bin/bash
# 5 configs × 10 seeds × 2 envs = 100 runs

cd /Users/saad/Desktop/regret_replay_RL
export PYTHONPATH="/Users/saad/Desktop/regret_replay_RL:$PYTHONPATH"

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

            python run_ablation.py \
                --config $config \
                --env $env \
                --seed $seed \
                --steps $STEPS
        done
    done
done

echo "Ablations complete: $(date)"
