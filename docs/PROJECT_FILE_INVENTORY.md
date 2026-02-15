# Project File Inventory — Regret Replay RL

Every file, its purpose, and where it is used. Layout updated: config, run, analyze, cleanup, docs, and logs live in dedicated folders.

---

## Root

| File | Purpose | Used by |
|------|---------|--------|
| **requirements.txt** | Python dependencies (gymnasium, torch, tensorboard, etc.) | `pip install -r requirements.txt` |

---

## config/ — Configuration

| File | Purpose | Used by |
|------|---------|--------|
| **__init__.py** | Central hyperparameters: BASELINE_CONFIG, RMER_CONFIG, PER_CONFIG, LFIW_CONFIG, TCE_CONFIG, HINDSIGHT_TD_CONFIG, ABLATION_CONFIGS, ENV_CONFIG | All trainers, buffers, run/ and analyze/ scripts (import as `config`) |
| **ablation_configs.py** | Ablation configs (lfiw_only, tce_only, htd_lfiw, htd_tce, lfiw_tce) and get_config/get_run_name | run/run_ablation.py, run/run_all_ablations.sh |
| **sensitivity_configs.py** | Sensitivity sweeps: PER α/β, priority temp, LFIW temp (12 configs); get_sensitivity_config | run/run_sensitivity.py, run/run_sensitivity_all.sh |

---

## run/ — Run experiments

| File | Purpose | Used by |
|------|---------|--------|
| **run_ablation.py** | Train one ablation (htd_lfiw, tce_only, etc.) with RMER; writes to runs/ablation_&lt;config&gt;_&lt;env&gt;_seed&lt;n&gt; | run_all_ablations.sh |
| **run_sensitivity.py** | Train one sensitivity config (per_alpha_0.4, temp_1.0, etc.); dispatches to PER or RMER trainer with run_suffix | run_sensitivity_all.sh, manual CLI |
| **run_all_ablations.sh** | 5 configs × 10 seeds × 2 envs = 100 runs; cd to root, set PYTHONPATH, call run_ablation.py | Manual / nohup |
| **run_sensitivity_all.sh** | 12 configs × 5 seeds × 1 env = 60 runs; calls run_sensitivity.py | Manual / nohup |
| **run_remern_experiments.sh** | 10 seeds × 2 envs ReMERN; calls phase3_rmer/train_remern.py | Manual / nohup |

---

## analyze/ — Result collection and plotting

| File | Purpose | Used by |
|------|---------|--------|
| **collect_all_results.py** | Collect main methods + ablations from runs/; writes results/complete_ALL_results.json. Run from project root. | plot_ablation_two_envs.py (reads JSON); run after experiments |
| **plot_ablation_two_envs.py** | Bar chart: CartPole and MountainCar (main + ablations) from results/complete_ALL_results.json; saves plots/ablation_two_environments.png. Run from project root. | Manual after collect_all_results.py |

---

## cleanup/ — Cleanup and verification

| File | Purpose | Used by |
|------|---------|--------|
| **clean_extra_runs.sh** | Move seed99 and sensitivity seeds 5–9 to archive_old/runs; cd to project root | Manual |
| **cleanup_project.sh** | Move legacy scripts, collectors, analysis, docs, results, logs to archive_old/; clean __pycache__, .pyc, etc.; cd to project root | Manual |
| **identify_old_runs.sh** | List runs with seed99 or sensitivity 5–9 (not in final 240/260); cd to runs/ | Manual |
| **verify_final_structure.sh** | Check project layout, phase file counts, runs breakdown, archive; cd to project root | Manual |

---

## docs/ — Documentation

| File | Purpose | Used by |
|------|---------|--------|
| **README.md** | How to run the code (install, single runs, batch, collect, plot, cleanup) | Users / developers |
| **IMPLEMENTATION.md** | Full project implementation: structure, phases, config, results, design notes | Reference |
| **README.txt** | Short summary; keep updating | Reference |
| **PROJECT_FILE_INVENTORY.md** | This file — list of files and purposes | Reference |

(Other docs such as PROJECT_FULL_SUMMARY.txt, ALMOST_THERE_BALANCE.md, etc. are moved to archive_old/docs/ by cleanup_project.sh.)

---

## logs/ — Batch run logs (optional)

| File | Purpose | Used by |
|------|---------|--------|
| **ablations_full.log** | stdout/stderr from run_all_ablations.sh | Debug / progress |
| **remern.log** | stdout/stderr from run_remern_experiments.sh | Debug / progress |
| **sensitivity.log** | stdout/stderr from run_sensitivity_all.sh | Debug / progress |

---

## phase1_baseline/ — DQN and baselines

| File | Purpose | Used by |
|------|---------|--------|
| **dqn.py** | DQN agent (Q-network, target network, update, epsilon decay, select_action); used by all DQN-based methods | train_baseline.py, train_per.py, RMER/ReMERN agents (via buffer’s agent reference) |
| **replay_buffer.py** | Uniform replay buffer (add, sample) | train_baseline.py |
| **per_buffer.py** | Prioritized replay (alpha, beta, TD-based priorities); add, sample, update_priorities | train_per.py |
| **train_baseline.py** | Train DQN with uniform replay; run name baseline_&lt;env&gt;_seed&lt;n&gt; | run_phase4.sh, run_mountaincar.sh, run_pilot.sh |
| **train_per.py** | Train DQN with PER; accepts config overrides and run_suffix for sensitivity; run name per_&lt;env&gt;_seed&lt;n&gt; or sensitivity_&lt;suffix&gt;_&lt;env&gt;_seed&lt;n&gt; | run_phase4.sh, run_mountaincar.sh, run_pilot.sh, run_sensitivity.py |

---

## phase2_components/ — RMER building blocks

| File | Purpose | Used by |
|------|---------|--------|
| **hindsight_td.py** | Hindsight TD error (post-update Q); compute_hindsight_td_error | rmer_buffer.py, remern_buffer.py (for priorities) |
| **lfiw.py** | LFIW: classifier for on-policy weight; temperature scaling; compute_on_policy_weights, update_step | rmer_buffer.py (LFIW weights in priority) |
| **tce.py** | TCE: temporal distance and Q-accuracy scores; DistanceTracker, TCEComputer | rmer_buffer.py (TCE accuracy in priority) |
| **error_network.py** | Error network: E(s,a) = max_a' Q(s,a') - Q(s,a); update, get_accuracy | remern_buffer.py (ReMERN priority component) |
| **test_components.py** | Unit tests for hindsight TD, LFIW, TCE | pytest / manual |

---

## phase3_rmer/ — RMER and ReMERN

| File | Purpose | Used by |
|------|---------|--------|
| **rmer_buffer.py** | RMER buffer: HTD × LFIW × TCE priorities; sample, update_priorities, update_lfiw, step; uses _lfiw_config_with_override for sensitivity | rmer_agent.py |
| **rmer_agent.py** | RMER agent: DQN + RMERBuffer; push, update(env_step), start/end_episode | train_rmer.py, run_ablation.py (via run_ablation’s agent creation) |
| **train_rmer.py** | RMERTrainer: full RMER training loop; optional run_suffix for sensitivity; run name rmer_&lt;env&gt;_seed&lt;n&gt;[ _noHTD_noLFIW_noTCE] or sensitivity_&lt;suffix&gt;_&lt;env&gt;_seed&lt;n&gt; | run_phase4.sh, run_mountaincar.sh, run_sensitivity.py |
| **remern_buffer.py** | ReMERN buffer: extends RMERBuffer; Error Network instead of TCE; update_priorities uses error_accuracy | remern_agent.py |
| **remern_agent.py** | ReMERN agent: DQN + ReMERNBuffer; update_batch returns (loss, htd_errors) for priority updates | train_remern.py |
| **train_remern.py** | Train ReMERN (Error Network variant); run name remern_&lt;env&gt;_seed&lt;n&gt; | run_remern_experiments.sh |

---

## utils/

| File | Purpose | Used by |
|------|---------|--------|
| **evaluation.py** | Evaluation helpers (e.g. run policy N episodes, return mean reward) | Possibly trainers or scripts (import if needed) |
| **visualization.py** | Plotting/visualization helpers | Possibly create_*_plots.py, analyze scripts |

---

## experiments/ (optional / legacy)

| File | Purpose | Used by |
|------|---------|--------|
| **compare_methods.py** | Compare methods (e.g. from logs/results) | Manual |
| **run_ablations.py** | Older ablation runner | Manual / legacy |
| **run_fixed_experiments.py** | Fixed-seed experiments | Manual / legacy |

---

## results/ — Collected outputs

| File | Purpose | Used by |
|------|---------|--------|
| **complete_ALL_results.json** | Main + ablation results (all methods, both envs, ablations); produced by analyze/collect_all_results.py | analyze/plot_ablation_two_envs.py |
| **remern_results.json** | ReMERN per-seed rewards (CartPole, MountainCar) | ReMERN summary / docs |
| **remern_summary.md** | Human-readable ReMERN summary | Reference |

Other result files (cartpole_results.json, mountaincar_results.json, comparison_*, ablation_*, etc.) may exist or be moved to archive_old/results/ by cleanup_project.sh.

---

## archive_old/

Holds archived scripts (run_phase4.sh, run_mountaincar.sh, run_pilot.sh, verify_experiments.sh), collectors, analysis scripts, extra docs, legacy results, and runs (e.g. seed99, sensitivity 5–9). Created/updated by cleanup/cleanup_project.sh and cleanup/clean_extra_runs.sh.

---

## plots/ (output directory)

Generated by analyze/plot_ablation_two_envs.py and (if present) archived plot scripts:  
**ablation_two_environments.png**, **learning_curves_comparison.png**, **cartpole_final_comparison.png**, **ranking_comparison.png**, **mountaincar_***, **rmer_results.png**, etc.  
**Used by:** Reports, README, papers.

---

## runs/ (output directory)

TensorBoard event files per run:  
**baseline_***, **per_***, **rmer_***, **remern_***, **ablation_***, **sensitivity_***.  
**Used by:** TensorBoard, analyze/collect_all_results.py.

---

## Quick dependency overview

- **config/** (with project root on PYTHONPATH) -> all trainers, buffers, run/ and analyze/ scripts.
- **phase1_baseline/dqn.py** -> baseline, PER, and (indirectly) RMER/ReMERN.
- **phase2_components/** -> **phase3_rmer/rmer_buffer.py** (and **remern_buffer.py** for error_network).
- **phase3_rmer/** -> **run/run_ablation.py**, **run/run_sensitivity.py**, **run/run_remern_experiments.sh**.
- **analyze/collect_all_results.py** -> **results/complete_ALL_results.json** -> **analyze/plot_ablation_two_envs.py** -> **plots/ablation_two_environments.png**.
