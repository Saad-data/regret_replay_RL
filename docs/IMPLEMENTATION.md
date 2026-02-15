# what we actually did (implementation)

rmer reimplementation + eval (Liu et al NeurIPS 21). cartpole + mountaincar.

---

## Overview

we did baseline (uniform), PER, full RMER (ReMERT with TCE), ReMERN (error net), and ablations. priority:

- **Hindsight TD error** (HTD)
- **Learned on-policy weight** (LFIW)
- **Q-accuracy term**: temporal coherence (TCE) in the paper; we also implement the **error network** variant (ReMERN)

Priority is computed as:

`priority = |hindsight_TD| × LFIW_weight × accuracy`

All experiments use DQN with 200,000 environment steps per run. We report mean and standard deviation of final evaluation reward over 10 seeds for main methods and ablations, and over 5 seeds for sensitivity runs.

---

## Project structure (what we built)

```
regret_replay_RL/
  requirements.txt         Python dependencies (at root)

  config/                  Configuration (project root on PYTHONPATH)
    __init__.py            Central hyperparameters (BASELINE, RMER, PER, LFIW, TCE, HTD, ENV, ABLATION_CONFIGS)
    ablation_configs.py    Ablation run configs (lfiw_only, tce_only, htd_lfiw, htd_tce, lfiw_tce)
    sensitivity_configs.py  Sensitivity sweeps (PER α/β, priority temp, LFIW temp — 12 configs)

  phase1_baseline/         DQN and baselines
    dqn.py                 DQN agent (used by all methods)
    replay_buffer.py       Uniform replay buffer
    per_buffer.py          Prioritized replay (PER)
    train_baseline.py      Uniform DQN training
    train_per.py            PER training

  phase2_components/       RMER building blocks
    hindsight_td.py        Hindsight TD error
    lfiw.py                On-policy weight (LFIW)
    tce.py                 Temporal coherence / accuracy (ReMERT)
    error_network.py       Error network (ReMERN)
    test_components.py     Component tests

  phase3_rmer/             RMER and ReMERN
    rmer_buffer.py         RMER buffer (ReMERT)
    rmer_agent.py          RMER agent
    train_rmer.py          ReMERT training
    remern_buffer.py       ReMERN buffer (error network)
    remern_agent.py        ReMERN agent
    train_remern.py        ReMERN training

  run/                     Run experiments
    run_ablation.py        Single ablation run
    run_sensitivity.py     Single sensitivity run
    run_all_ablations.sh   100 ablation runs
    run_sensitivity_all.sh 60 sensitivity runs
    run_remern_experiments.sh  20 ReMERN runs

  analyze/                 Collect results and plot
    collect_all_results.py   TensorBoard logs -> results/complete_ALL_results.json
    plot_ablation_two_envs.py results/complete_ALL_results.json -> plots/ablation_two_environments.png

  cleanup/                  Cleanup and verification
    clean_extra_runs.sh    Move seed99 / sensitivity 5–9 to archive_old/runs
    cleanup_project.sh     Archive legacy scripts/results to archive_old/
    identify_old_runs.sh   List runs not in final 240/260
    verify_final_structure.sh  Check project layout and run counts

  docs/                    Documentation
    README.md              How to run the code
    IMPLEMENTATION.md      This file — full implementation
    README.txt             Short summary (keep updating)

  logs/                    Batch run logs (optional)
  results/                 complete_ALL_results.json, remern_results.json, remern_summary.md
  plots/                   Output figures
  runs/                    TensorBoard logs (260 run dirs)
  archive_old/             Archived scripts and old runs
```

---

## What we implemented

### Phase 1 — Baseline

- **DQN agent** (`dqn.py`): Q-network, target network, epsilon-greedy, update step. Used by all DQN-based methods.
- **Uniform replay** (`replay_buffer.py`, `train_baseline.py`): standard DQN with uniform sampling.
- **PER** (`per_buffer.py`, `train_per.py`): prioritized experience replay (Schaul et al.); supports config overrides and `run_suffix` for sensitivity.

### Phase 2 — Components

- **Hindsight TD** (`hindsight_td.py`): hindsight TD error using post-update Q; used in RMER/ReMERN priority.
- **LFIW** (`lfiw.py`): classifier for on-policy weight; temperature scaling; used in RMER priority.
- **TCE** (`tce.py`): temporal distance and Q-accuracy (ReMERT); used in RMER buffer.
- **Error network** (`error_network.py`): E(s,a) = max_a' Q(s,a') − Q(s,a) for ReMERN; replaces TCE in priority.

### Phase 3 — RMER and ReMERN

- **RMER buffer** (`rmer_buffer.py`): priority = |HTD| × LFIW × TCE; sampling, priority updates, LFIW/TCE step.
- **RMER agent** (`rmer_agent.py`): DQN + RMERBuffer; used by `train_rmer.py` and `run_ablation.py`.
- **ReMERN buffer** (`remern_buffer.py`): same as RMER but error network instead of TCE for accuracy.
- **ReMERN agent** (`remern_agent.py`): DQN + ReMERNBuffer; used by `train_remern.py`.

### Config

- **config/__init__.py**: BASELINE_CONFIG, RMER_CONFIG, PER_CONFIG, LFIW_CONFIG, TCE_CONFIG, HINDSIGHT_TD_CONFIG, ENV_CONFIG, ABLATION_CONFIGS. All trainers and run scripts import `config`.
- **ablation_configs.py**: five ablation configs and `get_config` / `get_run_name` for `run_ablation.py`.
- **sensitivity_configs.py**: 12 sensitivity sweeps (PER alpha, PER beta, priority temperature, LFIW temperature) and `get_sensitivity_config` for `run_sensitivity.py`.

### Run / Analyze / Cleanup

- **run/**: scripts to run single ablation/sensitivity or full batches; they set project root and PYTHONPATH so `config` and phase modules resolve.
- **analyze/**: collect TensorBoard logs into `results/complete_ALL_results.json`; plot main methods + ablations for CartPole and MountainCar to `plots/ablation_two_environments.png`.
- **cleanup/**: move extra runs to archive, archive legacy files, list old runs, verify project structure.

---

## Project statistics

- **Python files:** phase1_baseline 5, phase2_components 5, phase3_rmer 6, config 3, run 2, analyze 2
- **Training runs:** 260 (20 baseline + 20 PER + 40 RMER + 20 ReMERN + 100 ablations + 60 sensitivity)
- **Seeds:** 10 per main method and per ablation; 5 per sensitivity config
- **Environments:** CartPole-v1, MountainCar-v0
- **Steps per run:** 200,000
- **Ablation configs:** 5 (LFIW-only, TCE-only, HTD+LFIW, HTD+TCE, LFIW+TCE)
- **Sensitivity configs:** 12 (PER alpha, PER beta, priority temperature, LFIW temperature)

---

## Main results (reported)

**CartPole-v1** (10 seeds, 200k steps). Success: final eval reward > 450.

| Method   | Mean (Std)   | Success |
|----------|--------------|---------|
| Uniform  | 500.0 (0.0)  | 10/10   |
| TD-only  | 458.7 (112.2)| 8/10    |
| ReMERN   | 435.0 (132.7)| 8/10    |
| Full-RMER| 395.4 (135.4)| 6/10    |
| PER      | 383.4 (138.9)| 5/10    |

**MountainCar-v0:** no run reached success threshold -110; uniform replay had best mean; full RMER/ReMERN concentrated at worst return.

**Ablations (CartPole-v1):** TCE-only and HTD+TCE among the best; LFIW+TCE lowest of the five.

---

## Implementation notes (design choices)

- **Epsilon decay:** by environment step, not gradient update count, so the schedule is consistent for a given `total_steps`.
- **RMER buffer:** uses stable transition IDs for TCE distance tracking; buffer indices are not used after wraparound.
- **TCE / time-based terms:** driven by environment step, not internal update count.
- **ReMERN:** error network predicts suboptimality; its output is used in the priority instead of the TCE term.

---

## Citation

Liu et al., *Regret Minimization Experience Replay in Off-Policy Reinforcement Learning*, NeurIPS 2021.

Repository: https://github.com/Saad-data/regret_replay_RL

---

## Contact

Syed Saad, hasan.2106512@studenti.uniroma1.it

Last updated: February 2026
