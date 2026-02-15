#!/bin/bash
# ReMERN experiments: 10 seeds × 2 environments (CartPole-v1, MountainCar-v0)

cd /Users/saad/Desktop/regret_replay_RL
export PYTHONPATH="/Users/saad/Desktop/regret_replay_RL:$PYTHONPATH"

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
echo "🎉 ReMERN experiments complete: $(date)"
