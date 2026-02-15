#!/bin/bash
# 20 remern runs (10 seeds x 2 envs). steps=20k in here - TODO might want 200k for paper

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"

STEPS=20000
SEEDS="0 1 2 3 4 5 6 7 8 9"

echo "========================================================================"
echo "ReMERN EXPERIMENTS"
echo "========================================================================"
echo "Start: $(date)"
echo ""

echo "CartPole-v1 (10 seeds)..."
for seed in $SEEDS; do
    echo "  Seed $seed: $(date +'%H:%M:%S')"
    python phase3_rmer/train_remern.py --env CartPole-v1 --seed $seed --steps $STEPS
done
echo "CartPole complete"
echo ""

echo "MountainCar-v0 (10 seeds)..."
for seed in $SEEDS; do
    echo "  Seed $seed: $(date +'%H:%M:%S')"
    python phase3_rmer/train_remern.py --env MountainCar-v0 --seed $seed --steps $STEPS
done
echo "MountainCar complete"

echo ""
echo "ReMERN experiments complete: $(date)"
