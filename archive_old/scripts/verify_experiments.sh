#!/usr/bin/env bash
# Verification checklist for CartPole experiments

cd "$(dirname "$0")"

echo "VERIFICATION CHECKLIST"
echo "======================"
echo ""

echo "1. Checking runs directory..."
run_count=$(ls runs/ 2>/dev/null | wc -l | tr -d ' ')
echo "   Found $run_count run directories"
echo "   Expected: 40 for CartPole (4 methods x 10 seeds)"
echo ""

echo "2. Checking results file..."
if [ -f "results/cartpole_results.json" ]; then
    echo "   OK: results/cartpole_results.json exists"
else
    echo "   MISSING: results/cartpole_results.json"
fi
echo ""

echo "3. Checking plots..."
if [ -f "plots/cartpole_comparison.png" ]; then
    echo "   OK: plots/cartpole_comparison.png exists"
else
    echo "   MISSING: plots/cartpole_comparison.png"
fi
echo ""

echo "4. To view in TensorBoard:"
echo "   tensorboard --logdir runs/ --port 6006"
echo "   Then open: http://localhost:6006"
echo ""

echo "Verification complete!"
