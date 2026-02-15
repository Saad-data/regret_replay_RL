#!/bin/bash
# MountainCar-v0 Full Experiment: 10 seeds × 4 methods × 200k steps

cd /Users/saad/Desktop/regret_replay_RL

ENV="MountainCar-v0"
STEPS=200000
SEEDS="0 1 2 3 4 5 6 7 8 9"

echo "========================================================================"
echo "MOUNTAINCAR-V0 EXPERIMENTS"
echo "========================================================================"
echo "Environment: $ENV"
echo "Training steps: $STEPS"
echo "Seeds: 10 (0-9)"
echo "Methods: 4 (Uniform, PER, TD-only, Full-RMER)"
echo "Total runs: 40"
echo ""
echo "Start: $(date)"
echo "Estimated: 4 hours (40 runs × 6 min)"
echo ""

current_run=0
total_runs=40

# Method 1: Uniform (10 seeds)
echo "========================================================================"
echo "METHOD 1/4: Uniform Baseline (10 seeds)"
echo "========================================================================"
for seed in $SEEDS; do
    current_run=$((current_run + 1))
    echo ""
    echo "[$current_run/$total_runs] Uniform seed $seed — $(date +'%H:%M:%S')"
    python phase1_baseline/train_baseline.py --env $ENV --seed $seed --steps $STEPS
    echo "✓ Uniform seed $seed complete — $(date +'%H:%M:%S')"
done
echo "✅ Uniform complete (10/40)"
echo ""

# Method 2: PER (10 seeds)
echo "========================================================================"
echo "METHOD 2/4: PER Baseline (10 seeds)"
echo "========================================================================"
for seed in $SEEDS; do
    current_run=$((current_run + 1))
    echo ""
    echo "[$current_run/$total_runs] PER seed $seed — $(date +'%H:%M:%S')"
    python phase1_baseline/train_per.py --env $ENV --seed $seed --steps $STEPS
    echo "✓ PER seed $seed complete — $(date +'%H:%M:%S')"
done
echo "✅ PER complete (20/40)"
echo ""

# Method 3: TD-only (10 seeds)
echo "========================================================================"
echo "METHOD 3/4: TD-only (Hindsight TD) (10 seeds)"
echo "========================================================================"
for seed in $SEEDS; do
    current_run=$((current_run + 1))
    echo ""
    echo "[$current_run/$total_runs] TD-only seed $seed — $(date +'%H:%M:%S')"
    python phase3_rmer/train_rmer.py --env $ENV --seed $seed --steps $STEPS --no-lfiw --no-tce
    echo "✓ TD-only seed $seed complete — $(date +'%H:%M:%S')"
done
echo "✅ TD-only complete (30/40)"
echo ""

# Method 4: Full RMER (10 seeds)
echo "========================================================================"
echo "METHOD 4/4: Full RMER (HTD + LFIW + TCE) (10 seeds)"
echo "========================================================================"
for seed in $SEEDS; do
    current_run=$((current_run + 1))
    echo ""
    echo "[$current_run/$total_runs] Full RMER seed $seed — $(date +'%H:%M:%S')"
    python phase3_rmer/train_rmer.py --env $ENV --seed $seed --steps $STEPS
    echo "✓ Full RMER seed $seed complete — $(date +'%H:%M:%S')"
done
echo "✅ Full RMER complete (40/40)"
echo ""

echo "========================================================================"
echo "🎉 MOUNTAINCAR EXPERIMENTS COMPLETE!"
echo "========================================================================"
echo "End: $(date)"
echo "Total: 40 runs (10 seeds × 4 methods)"
echo ""
echo "Next: python collect_mountaincar_results.py"
echo "      python create_mountaincar_plots.py"
echo "========================================================================"
