#!/usr/bin/env bash
# Phase 4: Remaining 7 seeds (3-9) for all 4 methods.
# Usage: bash run_phase4.sh [STEPS]
#   STEPS defaults to 200000 if not given.
# Example: nohup bash run_phase4.sh 200000 > phase4.log 2>&1 &

set -e
cd /Users/saad/Desktop/regret_replay_RL

STEPS="${1:-200000}"
ENV="CartPole-v1"
REMAINING_SEEDS="3 4 5 6 7 8 9"

echo "========================================================================"
echo "PHASE 4: Seeds 3-9, ${STEPS} steps"
echo "========================================================================"
echo "Start: $(date)"
echo "Runs: 4 methods × 7 seeds = 28"
echo ""

total_runs=28
current_run=0

# Method 1: Uniform (7 seeds)
echo "========================================================================"
echo "METHOD 1/4: Uniform Baseline (seeds $REMAINING_SEEDS)"
echo "========================================================================"
for seed in $REMAINING_SEEDS; do
    current_run=$((current_run + 1))
    echo ""
    echo "[$current_run/$total_runs] Uniform seed $seed — $(date +'%H:%M:%S')"
    python phase1_baseline/train_baseline.py --env $ENV --seed $seed --steps $STEPS
    echo "✓ Uniform seed $seed complete: $(date +'%H:%M:%S')"
done
echo "✅ Uniform complete (7/28)"
echo ""

# Method 2: PER (7 seeds)
echo "========================================================================"
echo "METHOD 2/4: PER Baseline (seeds $REMAINING_SEEDS)"
echo "========================================================================"
for seed in $REMAINING_SEEDS; do
    current_run=$((current_run + 1))
    echo ""
    echo "[$current_run/$total_runs] PER seed $seed — $(date +'%H:%M:%S')"
    python phase1_baseline/train_per.py --env $ENV --seed $seed --steps $STEPS
    echo "✓ PER seed $seed complete: $(date +'%H:%M:%S')"
done
echo "✅ PER complete (14/28)"
echo ""

# Method 3: TD-only (7 seeds)
echo "========================================================================"
echo "METHOD 3/4: TD-only (seeds $REMAINING_SEEDS)"
echo "========================================================================"
for seed in $REMAINING_SEEDS; do
    current_run=$((current_run + 1))
    echo ""
    echo "[$current_run/$total_runs] TD-only seed $seed — $(date +'%H:%M:%S')"
    python phase3_rmer/train_rmer.py --env $ENV --seed $seed --steps $STEPS --no-lfiw --no-tce
    echo "✓ TD-only seed $seed complete: $(date +'%H:%M:%S')"
done
echo "✅ TD-only complete (21/28)"
echo ""

# Method 4: Full RMER (7 seeds)
echo "========================================================================"
echo "METHOD 4/4: Full RMER (seeds $REMAINING_SEEDS)"
echo "========================================================================"
for seed in $REMAINING_SEEDS; do
    current_run=$((current_run + 1))
    echo ""
    echo "[$current_run/$total_runs] Full RMER seed $seed — $(date +'%H:%M:%S')"
    python phase3_rmer/train_rmer.py --env $ENV --seed $seed --steps $STEPS
    echo "✓ Full RMER seed $seed complete: $(date +'%H:%M:%S')"
done
echo "✅ Full RMER complete (28/28)"
echo ""

echo "========================================================================"
echo "🎉 PHASE 4 COMPLETE!"
echo "========================================================================"
echo "End: $(date)"
echo ""
echo "Next: python collect_cartpole_results.py   # or python collect_all_results.py"
echo "       python create_plots.py              # or python create_final_plots.py"
echo "========================================================================"
