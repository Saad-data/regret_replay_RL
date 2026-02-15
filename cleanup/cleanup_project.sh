#!/bin/bash
# move old scripts/logs/collectors into archive_old. run from root.

set -e
cd "$(dirname "$0")/.."

echo "Creating archive structure..."
mkdir -p archive_old/{scripts,collectors,analysis,docs,results,logs,runs}

echo "Moving early experiment scripts..."
mv run_pilot.sh run_phase4.sh run_mountaincar.sh verify_experiments.sh archive_old/scripts/ 2>/dev/null || true

echo "Moving logs..."
mv pilot.log phase4.log mountaincar.log archive_old/logs/ 2>/dev/null || true

echo "Moving collectors..."
mv collect_cartpole_results.py collect_mountaincar_results.py collect_remern_results.py \
   collect_full_rmer_results.py collect_td_only_results.py archive_old/collectors/ 2>/dev/null || true

echo "Moving analysis scripts..."
mv plot_learning_curves.py create_final_plots.py create_mountaincar_plots.py create_plots.py \
   analyze_pilot.py compare_seeds.py diagnose_curves.py check_epsilon_curves.py \
   final_comparison.py archive_old/analysis/ 2>/dev/null || true

echo "Moving docs..."
mv PROJECT_FULL_SUMMARY.txt ALMOST_THERE_BALANCE.md DEEP_ANALYSIS_EPSILON_AND_RESULTS.md \
   INFORMATION_CHECKLIST.md RESEARCH_CONTRIBUTIONS.md archive_old/docs/ 2>/dev/null || true

echo "Moving legacy folders..."
mv experiments archive_old/ 2>/dev/null || true
mv fixed archive_old/ 2>/dev/null || true
mv utils archive_old/ 2>/dev/null || true

echo "Moving old results (keep complete_ALL_results.json, remern_*)..."
cd results
mv cartpole_results.json ../archive_old/results/ 2>/dev/null || true
mv mountaincar_results.json ../archive_old/results/ 2>/dev/null || true
mv cartpole_final_results.json ../archive_old/results/ 2>/dev/null || true
mv comparison_* ../archive_old/results/ 2>/dev/null || true
mv ablation_* ../archive_old/results/ 2>/dev/null || true
cd ..

echo "Cleaning cache..."
find . -type d -name "__pycache__" -not -path "./archive_old*" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -not -path "./archive_old*" -delete 2>/dev/null || true
find . -type f -name ".DS_Store" -not -path "./archive_old*" -delete 2>/dev/null || true
find . -type f \( -name "*.swp" -o -name "*.swo" \) -not -path "./archive_old*" -delete 2>/dev/null || true
find . -type d -name ".ipynb_checkpoints" -not -path "./archive_old*" -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "Cleanup complete!"
echo "Project has clean structure. Archive is in: archive_old/"
