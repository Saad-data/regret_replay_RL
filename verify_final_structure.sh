#!/bin/bash
# Final verification of clean project structure

cd "$(dirname "$0")"

echo "==================================================================="
echo "FINAL PROJECT STRUCTURE VERIFICATION"
echo "==================================================================="
echo ""

echo "1. ROOT FILES:"
echo "   Python scripts: $(ls -1 *.py 2>/dev/null | wc -l)"
echo "   Shell scripts: $(ls -1 *.sh 2>/dev/null | wc -l)"
echo "   Config/docs: $(ls -1 *.md *.txt 2>/dev/null | wc -l)"
echo "   Logs: $(ls -1 *.log 2>/dev/null | wc -l)"

echo ""
echo "2. IMPLEMENTATION FOLDERS:"
ls -d phase*/ 2>/dev/null

echo ""
echo "3. PHASE 1 (Baseline): $(ls -1 phase1_baseline/*.py 2>/dev/null | wc -l) files (expected 5)"
echo "4. PHASE 2 (Components): $(ls -1 phase2_components/*.py 2>/dev/null | wc -l) files (expected 4)"
echo "5. PHASE 3 (RMER): $(ls -1 phase3_rmer/*.py 2>/dev/null | wc -l) files (expected 6)"

echo ""
echo "6. RESULTS:"
ls -1 results/*.json results/*.md 2>/dev/null
echo "   Expected: complete_ALL_results.json, remern_results.json, remern_summary.md"

echo ""
echo "7. PLOTS: $(ls -1 plots/*.png 2>/dev/null | wc -l) .png files"

echo ""
echo "8. RUNS BREAKDOWN:"
echo "   Baseline:    $(ls -d runs/baseline_* 2>/dev/null | wc -l) (expected 20)"
echo "   PER:         $(ls -d runs/per_* 2>/dev/null | wc -l) (expected 20)"
echo "   RMER:        $(ls -d runs/rmer_* 2>/dev/null | wc -l) (expected 40)"
echo "   ReMERN:      $(ls -d runs/remern_* 2>/dev/null | wc -l) (expected 20)"
echo "   Ablations:   $(ls -d runs/ablation_* 2>/dev/null | wc -l) (expected 100)"
echo "   Sensitivity: $(ls -d runs/sensitivity_* 2>/dev/null | wc -l) (expected 60)"
echo "   TOTAL:       $(ls -d runs/*/ 2>/dev/null | wc -l) (expected 260 after removing seed99)"

echo ""
echo "9. ARCHIVE:"
ls archive_old/ 2>/dev/null

echo ""
echo "==================================================================="
echo "VERIFICATION COMPLETE"
echo "==================================================================="
