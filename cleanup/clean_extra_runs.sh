#!/bin/bash
# dump seed99 and sensitivity 5-9 into archive so we only have final 240/260

cd "$(dirname "$0")/.."
mkdir -p archive_old/runs

echo "Moving extra runs to archive_old/runs/..."
echo ""

moved=0

for run in runs/*_seed99*; do
    if [ -d "$run" ]; then
        echo "  $run"
        mv "$run" archive_old/runs/
        ((moved++)) || true
    fi
done

for run in runs/sensitivity_*_seed[5-9]; do
    if [ -d "$run" ]; then
        echo "  $run"
        mv "$run" archive_old/runs/
        ((moved++)) || true
    fi
done

echo ""
echo "Moved $moved runs to archive_old/runs/"
echo "Remaining runs: $(ls -d runs/*/ 2>/dev/null | wc -l)"
