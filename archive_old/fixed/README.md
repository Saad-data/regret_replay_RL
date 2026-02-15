# Fixed logic experiments

Results from re-running RMER with the corrected implementation:

- **Buffer:** Circular buffer + correct transition-ID mapping for TCE distances
- **TCE:** Progress uses env steps (not update steps)
- **Agent:** `total_steps` set from actual training length
- **LFIW:** Fast/slow split by age when buffer is full

## How to run

```bash
python experiments/run_fixed_experiments.py --env LunarLander-v2 --device cpu
```

## Layout

- `fixed/runs/` — TensorBoard logs (one dir per config per seed)
- `fixed/checkpoints/` — Saved model checkpoints (if save_freq reached)
- `fixed/plots/` — Per-run plot from last trainer (optional)
- `fixed/results/` — Aggregated results and comparison
  - `fixed_results_<timestamp>.json` — Mean/std and per-seed rewards for td_only and full_rmer
  - `plots/ablation_study.png` — Bar plot
  - `plots/learning_curves.png` — Learning curves

## Config

- **Steps:** 200,000 per run
- **Seeds:** 0–9 (10 seeds)
- **Configs:** td_only, full_rmer
