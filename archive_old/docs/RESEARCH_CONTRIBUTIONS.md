# Research Contributions (RMER Replication Study)

## Summary

This document captures the scientific contributions from the RMER (Regret Minimization Experience Replay) replication on LunarLander-v2, including discovery and fix of a configuration bug and the resulting empirical findings.

---

## 1. Configuration Bug Discovery & Fix

- **Identified** a subtle semantic mismatch: `epsilon_decay_steps` in config was intended as **environment steps**, but the DQN implementation decayed epsilon using **gradient update count**.
- **Diagnosed** via epsilon curve analysis (TensorBoard + scripts): seeds 0–4 reached ε = 0.05 by ~41k env steps; seeds 5–9 stayed at ε ≈ 0.53 at 200k steps.
- **Fixed** with env-step decay: added `decay_epsilon_by_env_step(env_step)` in the DQN, trainers pass `step` every env step, removed update-based decay from `update()`.
- **Validated** with a full re-run of seeds 0–9 (td_only and full_rmer) under identical, fixed code and config (`epsilon_decay_steps: 20_000` env steps).

**Technical specification:** See `DEEP_ANALYSIS_EPSILON_AND_RESULTS.md` for full call flow and before/after comparison.

---

## 2. Methodological Insights

- **Sampling and power:** Demonstrated impact of unbalanced seeds (5 vs 10) on statistical conclusions; balanced 10 vs 10 comparison is essential.
- **Diagnostics:** Epsilon curves, loss, and eval reward over time were used to pinpoint when and why runs diverged (exploration schedule vs learning).
- **Configuration validation:** Showed that config parameters must be validated against the actual code path (env steps vs update steps).
- **Evaluation protocol:** Established a clear pipeline: fixed config → re-run all seeds → collect from TensorBoard → final comparison with significance tests and effect size (Cohen’s d).

---

## 3. Empirical Finding

**After the fix (valid comparison, 10 seeds each):**

- **td_only (Hindsight TD only):** mean −18.70 ± 59.10  
- **full_rmer (TD + LFIW + TCE):** mean −33.72 ± 48.46  
- **Difference:** +15.01 points in favor of td_only (not statistically significant).  
- **p-value:** 0.54 (no significant difference between methods).  
- **Cohen’s d:** −0.28 (small effect, td_only slightly better on average).  
- **Variance:** full_rmer has lower standard deviation (48.46 vs 59.10).

**Interpretation:**

- Remaining differences are **algorithm and environment variance**, not the epsilon bug.
- **Null result:** Under this setup (LunarLander-v2, 200k steps, fixed exploration), we do not find a significant advantage of full RMER over td_only; td_only is slightly better on mean, full_rmer is more stable.
- This **challenges** any universal claim that “RMER always improves over baseline” and highlights **environment- and setup-specific** behavior.

---

## References in Repo

- **Deep technical analysis:** `DEEP_ANALYSIS_EPSILON_AND_RESULTS.md`  
- **Information checklist (timestamps, curves):** `INFORMATION_CHECKLIST.md`  
- **Config:** `config.py` (`epsilon_decay_steps`, env-step decay)  
- **Fix locations:** `phase1_baseline/dqn.py`, `phase3_rmer/train_rmer.py`, `phase1_baseline/train_baseline.py`
