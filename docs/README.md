# how to run the code

run everything from project root (`regret_replay_RL/`). set PYTHONPATH so config loads.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Single runs (one method, one seed)

need PYTHONPATH=. for config import.

**Baseline (uniform replay):**
```bash
PYTHONPATH=. python phase1_baseline/train_baseline.py --env CartPole-v1 --seed 0 --steps 200000
```

**PER (prioritized experience replay):**
```bash
PYTHONPATH=. python phase1_baseline/train_per.py --env CartPole-v1 --seed 0 --steps 200000
```

**Full RMER (ReMERT — TCE):**
```bash
PYTHONPATH=. python phase3_rmer/train_rmer.py --env CartPole-v1 --seed 0 --steps 200000
```

**ReMERN (error network):**
```bash
PYTHONPATH=. python phase3_rmer/train_remern.py --env CartPole-v1 --seed 0 --steps 200000
```

**One ablation (e.g. htd_lfiw, tce_only, lfiw_only, htd_tce, lfiw_tce):**
```bash
PYTHONPATH=. python run/run_ablation.py --config htd_lfiw --env CartPole-v1 --seed 0 --steps 200000
```

**One sensitivity config (e.g. per_alpha_0.4, temp_1.0):**
```bash
PYTHONPATH=. python run/run_sensitivity.py --config per_alpha_0.4 --env CartPole-v1 --seed 0 --steps 200000
```

Use `--env MountainCar-v0` for MountainCar. Logs go to `runs/`.

## 3. Batch experiments

scripts cd to root and set PYTHONPATH. run from root:

```bash
./run/run_all_ablations.sh       # 100 runs (5 configs × 10 seeds × 2 envs)
./run/run_sensitivity_all.sh    # 60 runs (12 configs × 5 seeds × CartPole)
./run/run_remern_experiments.sh # 20 ReMERN runs (10 seeds × 2 envs)
```

optional: redirect to logs/ablations_full.log 2>&1 or whatever.

## 4. Collect results and plot

After runs finish, aggregate TensorBoard logs and plot:

```bash
python analyze/collect_all_results.py      # -> results/complete_ALL_results.json
python analyze/plot_ablation_two_envs.py    # -> plots/ablation_two_environments.png
```

needs tensorboard + matplotlib (in requirements).

## 5. Cleanup and verification (optional)

```bash
./cleanup/clean_extra_runs.sh      # move seed99 / sensitivity 5–9 to archive_old/runs
./cleanup/cleanup_project.sh        # archive legacy scripts/results to archive_old/
./cleanup/identify_old_runs.sh      # list runs not in final 240/260
./cleanup/verify_final_structure.sh # check layout and run counts
```

---

full writeup: [IMPLEMENTATION.md](IMPLEMENTATION.md). quick ref: [README.txt](README.txt).
