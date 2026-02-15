#!/usr/bin/env bash
# Phase 2: Pilot study — 4 methods × 3 seeds × 500k steps (~2.5 hours)

set -e
cd /Users/saad/Desktop/regret_replay_RL

ENV="CartPole-v1"
STEPS=500000
PILOT_SEEDS="0 1 2"

echo "========================================================================"
echo "PILOT STUDY: 500k steps, 3 seeds"
echo "========================================================================"
echo "Start: $(date)"
echo "Estimated: 2.5 hours (4 methods × 3 seeds × ~12 min)"
echo ""
echo "Progress: 12 runs total"
echo ""

total_runs=12
current_run=0

# Method 1: Uniform (3 seeds)
echo "========================================================================"
echo "METHOD 1/4: Uniform Baseline (3 seeds)"
echo "========================================================================"
for seed in $PILOT_SEEDS; do
    current_run=$((current_run + 1))
    echo ""
    echo "[$current_run/$total_runs] Uniform seed $seed — Started: $(date +'%H:%M:%S')"
    python phase1_baseline/train_baseline.py --env $ENV --seed $seed --steps $STEPS
    echo "✓ Uniform seed $seed complete: $(date +'%H:%M:%S')"
done
echo "✅ Uniform pilot complete (3/12 runs) — $(date +'%H:%M:%S')"
echo ""

# Method 2: PER (3 seeds)
echo "========================================================================"
echo "METHOD 2/4: PER Baseline (3 seeds)"
echo "========================================================================"
for seed in $PILOT_SEEDS; do
    current_run=$((current_run + 1))
    echo ""
    echo "[$current_run/$total_runs] PER seed $seed — Started: $(date +'%H:%M:%S')"
    python phase1_baseline/train_per.py --env $ENV --seed $seed --steps $STEPS
    echo "✓ PER seed $seed complete: $(date +'%H:%M:%S')"
done
echo "✅ PER pilot complete (6/12 runs) — $(date +'%H:%M:%S')"
echo ""

# Method 3: TD-only (3 seeds)
echo "========================================================================"
echo "METHOD 3/4: TD-only (Hindsight TD) (3 seeds)"
echo "========================================================================"
for seed in $PILOT_SEEDS; do
    current_run=$((current_run + 1))
    echo ""
    echo "[$current_run/$total_runs] TD-only seed $seed — Started: $(date +'%H:%M:%S')"
    python phase3_rmer/train_rmer.py --env $ENV --seed $seed --steps $STEPS --no-lfiw --no-tce
    echo "✓ TD-only seed $seed complete: $(date +'%H:%M:%S')"
done
echo "✅ TD-only pilot complete (9/12 runs) — $(date +'%H:%M:%S')"
echo ""

# Method 4: Full RMER (3 seeds)
echo "========================================================================"
echo "METHOD 4/4: Full RMER (HTD + LFIW + TCE) (3 seeds)"
echo "========================================================================"
for seed in $PILOT_SEEDS; do
    current_run=$((current_run + 1))
    echo ""
    echo "[$current_run/$total_runs] Full RMER seed $seed — Started: $(date +'%H:%M:%S')"
    python phase3_rmer/train_rmer.py --env $ENV --seed $seed --steps $STEPS
    echo "✓ Full RMER seed $seed complete: $(date +'%H:%M:%S')"
done
echo "✅ Full RMER pilot complete (12/12 runs)"
echo ""

echo "========================================================================"
echo "🎉 PILOT STUDY COMPLETE!"
echo "========================================================================"
echo "End: $(date)"
echo ""
echo "Next: python analyze_pilot.py"
echo "       tensorboard --logdir runs/ --port 6006"
echo "========================================================================"
