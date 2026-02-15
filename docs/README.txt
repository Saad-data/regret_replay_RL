Regret Minimization Experience Replay (RMER) - reimplementation
================================================================
Liu et al NeurIPS 21. CartPole + MountainCar. Baselines, PER, RMER, ReMERN, ablations, sensitivity.

docs (3 files, keep this one updated)
-------------------------------------
  README.md         how to run (install, single/batch, collect, plot)
  IMPLEMENTATION.md full writeup of what we built
  README.txt        this file - short summary

OVERVIEW
--------
  priority = |hindsight_TD| * LFIW_weight * accuracy
  ReMERT = TCE for accuracy; ReMERN = error network for accuracy.
  200k steps/run; 10 seeds main/ablations, 5 seeds sensitivity.

PROJECT STRUCTURE
-----------------
  requirements.txt (root)
  config/          __init__.py, ablation_configs.py, sensitivity_configs.py
  phase1_baseline/  DQN, uniform replay, PER (dqn, replay_buffer, per_buffer, train_baseline, train_per)
  phase2_components/ HTD, LFIW, TCE, error_network, test_components (5 files)
  phase3_rmer/     rmer_buffer, rmer_agent, train_rmer, remern_buffer, remern_agent, train_remern
  run/             run_ablation.py, run_sensitivity.py, run_all_ablations.sh, run_sensitivity_all.sh, run_remern_experiments.sh
  analyze/         collect_all_results.py, plot_ablation_two_envs.py
  cleanup/         clean_extra_runs.sh, cleanup_project.sh, identify_old_runs.sh, verify_final_structure.sh
  docs/            README.md, IMPLEMENTATION.md, README.txt
  logs/            ablations_full.log, remern.log, sensitivity.log
  results/         complete_ALL_results.json, remern_results.json, remern_summary.md
  plots/           ablation_two_environments.png, etc.
  runs/            TensorBoard logs (260 dirs)
  archive_old/     Archived scripts and old runs

STATS
-----
  Python: phase1 5, phase2 5, phase3 6, config 3, run 2, analyze 2
  Runs: 260 (20 baseline + 20 PER + 40 RMER + 20 ReMERN + 100 ablations + 60 sensitivity)
  Envs: CartPole-v1, MountainCar-v0. Steps: 200k. Ablations: 5. Sensitivity: 12 configs.

HOW TO RUN (summary)
--------------------
  pip install -r requirements.txt
  PYTHONPATH=. python phase1_baseline/train_baseline.py --env CartPole-v1 --seed 0 --steps 200000
  ./run/run_all_ablations.sh  ;  ./run/run_sensitivity_all.sh  ;  ./run/run_remern_experiments.sh
  python analyze/collect_all_results.py  ;  python analyze/plot_ablation_two_envs.py
  ./cleanup/verify_final_structure.sh
  See README.md for full run instructions.

MAIN RESULTS (CartPole, 10 seeds)
----------------------------------
  Uniform 500.0 (0.0) 10/10  |  TD-only 458.7 8/10  |  ReMERN 435.0 8/10  |  Full-RMER 395.4 6/10  |  PER 383.4 5/10

ABLATION (CartPole)
-------------------
  TCE-only 490.3 9/10  |  HTD+TCE 478.1 9/10  |  LFIW-only 474.8 9/10  |  HTD+LFIW 474.7 9/10  |  LFIW+TCE 445.9 8/10

CITATION / CONTACT
------------------
  Liu et al., Regret Minimization Experience Replay in Off-Policy Reinforcement Learning, NeurIPS 2021.
  https://github.com/Saad-data/regret_replay_RL
  Syed Saad, hasan.2106512@studenti.uniroma1.it

Last updated: February 2026
