#!/bin/bash
# Identify runs that are NOT part of final 240 (seeds 0-9 only, sensitivity seeds 0-4)

cd "$(dirname "$0")/runs" || exit 1

echo "==================================================================="
echo "IDENTIFYING OLD/EXTRA RUNS"
echo "==================================================================="
echo ""

# Expected: seeds 0-9 only (no seed99), sensitivity seeds 0-4 only
# So "old" = anything with seed99, or sensitivity with seed 5-9
echo "Runs with seed 99 (test runs - NOT in final 240):"
ls -d *_seed99* 2>/dev/null || echo "  (none)"
echo ""

echo "Sensitivity runs with seed 5-9 (should only be 0-4):"
ls -d sensitivity_*_seed[5-9] 2>/dev/null || echo "  (none)"
echo ""

echo "All other run types use seeds 0-9."
echo "Total current runs: $(ls -d */ 2>/dev/null | wc -l)"
echo "After moving seed99 (and sensitivity 5-9 if any): expected 240 or 260."
echo ""
echo "-------------------------------------------------------------------"
echo "Summary of OLD runs to move to archive_old/runs/:"
ls -d *_seed99* 2>/dev/null
ls -d sensitivity_*_seed[5-9] 2>/dev/null
echo "-------------------------------------------------------------------"
