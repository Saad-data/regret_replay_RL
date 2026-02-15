# Deep Internal Analysis: Epsilon Bug, Fix, and Results

## 1. The Exact Problem (Root Cause)

### 1.1 Semantic Mismatch

**Config (what humans expect):**
- `epsilon_decay_steps: 100_000` was written and read as **“decay over 100,000 environment steps.”**
- So by step 100,000, epsilon should be 0.05.

**Code (what actually happened):**
- Epsilon was decayed inside `DQNAgent.update()` using **`self.train_step`** (number of **gradient updates**), not environment steps.
- So `epsilon_decay_steps` was effectively interpreted as **“number of gradient updates over which to decay.”**

**Relationship:**
- `train_freq = 4` → one gradient update every 4 env steps.
- So **200,000 env steps ⇒ 50,000 updates.**

So with `epsilon_decay_steps = 100_000` (interpreted as 100k **updates**):
- After 200k env steps you have only 50k updates.
- Epsilon = 1 − (1 − 0.05) × (50,000 / 100,000) = **0.525** (stuck near 0.53).

That is the **exact** cause of seeds 5–9 never reaching 0.05.

---

## 2. Where the Bug Lived (Exact Locations)

### 2.1 DQN: decay was tied to gradient updates

**File:** `phase1_baseline/dqn.py`

**Original logic (bug):**
- **Line 216 (old):** Inside `update()`, after `self.train_step += 1`, the code called **`self.update_epsilon()`**.
- **Lines 165–171:** `update_epsilon()` does:
  ```python
  self.epsilon = max(
      self.epsilon_end,
      self.epsilon_start - (self.epsilon_start - self.epsilon_end) *
      self.train_step / self.epsilon_decay_steps   # ← train_step = # of updates
  )
  ```
- So epsilon was **only** updated when `update()` ran (every `train_freq` env steps), and the divisor was `epsilon_decay_steps` in **update count**.

**Why this is wrong:**
- The trainer loop is driven by **env steps** (`for step in range(total_steps)`).
- The agent never saw `step`; it only saw its own `train_step` (updates).
- So the same config value was used with the wrong “clock”: config intended as env steps, code used as update steps.

### 2.2 Trainers: env step never used for epsilon

**Files:** `phase3_rmer/train_rmer.py`, `phase1_baseline/train_baseline.py`

**Original logic (missing piece):**
- The training loop has `step` (env step) but **did not** pass it to the agent.
- So the agent had no way to decay epsilon by env step; it could only use `train_step` inside `update()`.

**Exact gap:**
- No call like `agent.decay_epsilon_by_env_step(step)` (or equivalent) existed.
- So epsilon schedule was fully determined inside DQN by update count.

---

## 3. Why Seeds 0–4 Reached 0.05 (Different Code Path)

Evidence from TensorBoard:
- **Seed 0:** Epsilon reached 0.05 by **~41,000 env steps** (decay over ~40k **env** steps).
- **Seed 5:** Epsilon never reached 0.05 (ended at 0.53, i.e. **update**-based decay).

So seeds 0–4 were **not** run with the same code path as seeds 5–9. Possibilities consistent with the data:

1. **Different code version:** An older (or different) version that decayed epsilon using **env step** (e.g. received `step` from the trainer and used it for decay). Then `epsilon_decay_steps` in config could be 40k–50k **env** steps and you’d see 0.05 by ~41k.
2. **Different config at run time:** Same code (update-based) but with `epsilon_decay_steps = 50_000` (updates). Then 50k updates would complete decay—but 50k updates happen at 200k env steps, not at 41k. So at 41k env steps you’d have ~10k updates and epsilon ≈ 0.81, which does **not** match the curve. So for seed 0 to hit 0.05 by 41k **steps**, decay had to be driven by **env steps**, not updates.

**Conclusion:** Seeds 0–4 were run with **env-step** decay (different code or different code path). Seeds 5–9 were run with **update-step** decay and `epsilon_decay_steps = 100_000` (updates). The “exact problem” for 5–9 was: **only** update-based decay existed in the code you ran, and the trainer never passed env step into the agent.

---

## 4. The Fix (What Changed and Why It Works)

### 4.1 New decay by env step in DQN

**File:** `phase1_baseline/dqn.py`

**Added (lines 173–179):**
```python
def decay_epsilon_by_env_step(self, env_step):
    """Decay epsilon linearly by environment step. Call from trainer each step."""
    self.epsilon = max(
        self.epsilon_end,
        self.epsilon_start - (self.epsilon_start - self.epsilon_end) *
        min(1.0, env_step / self.epsilon_decay_steps)
    )
```

- Uses **env_step** and **epsilon_decay_steps** in the same units (env steps).
- So when config says `epsilon_decay_steps: 20_000`, epsilon reaches 0.05 at **step 20,000**, as intended.

**Removed from `update()`:**
- The call to `self.update_epsilon()` was removed so epsilon is **no longer** driven by `train_step`. It is driven only by the trainer’s call to `decay_epsilon_by_env_step(step)`.

So the **same** config key now means the same thing in code and in your head: “steps” = env steps.

### 4.2 Trainers pass env step every step

**File:** `phase3_rmer/train_rmer.py` (lines 99–100):
```python
for step in range(total_steps):
    self.agent.dqn_agent.decay_epsilon_by_env_step(step)
    action = self.agent.select_action(state)
```

**File:** `phase1_baseline/train_baseline.py` (lines 96–97):
```python
for step in range(total_steps):
    self.agent.decay_epsilon_by_env_step(step)
    action = self.agent.select_action(state)
```

- **Every** env step, the trainer passes the current **step** to the agent.
- The agent then uses that for epsilon, so the schedule is deterministic and consistent across runs and methods.

### 4.3 Config meaning and value

**File:** `config.py`

- **Value:** `epsilon_decay_steps: 20_000` (changed from 100_000).
- **Meaning:** “Epsilon goes from 1.0 to 0.05 over the first 20,000 **env steps**.”
- So by step 20k, epsilon = 0.05; from 20k to 200k you have pure exploitation. This matches the intended “~10% of training” schedule.

**Why this fixes the “problem”:**
- **Before:** Epsilon was driven by update count; 200k steps ⇒ 50k updates ⇒ with 100k “steps” (interpreted as updates) epsilon stayed at ~0.53.
- **After:** Epsilon is driven by env step; at 20k steps epsilon = 0.05; all seeds and all runs see the same schedule. No more silent mismatch between config and implementation.

---

## 5. Call Flow: Before vs After

### Before (bug)

1. Trainer: `step = 0, 1, 2, ...` (env steps).
2. Every `train_freq` steps, trainer calls `agent.update()`.
3. In `update()`: `self.train_step += 1` then `self.update_epsilon()`.
4. `update_epsilon()`: `epsilon = f(train_step, epsilon_decay_steps)` with both in **updates**.
5. So after 200k env steps: `train_step = 50_000`, `epsilon_decay_steps = 100_000` → epsilon = 0.525.

Epsilon was **never** a function of env `step`; only of update count.

### After (fix)

1. Trainer: `step = 0, 1, 2, ...` (env steps).
2. **Every** step: `agent.decay_epsilon_by_env_step(step)` so `epsilon = f(step, epsilon_decay_steps)` in **env steps**.
3. Then action is selected with that epsilon.
4. `update()` no longer touches epsilon.

So epsilon is a deterministic function of env step and config; 20k steps ⇒ 0.05.

---

## 6. Why Results Look Like They Do After the Fix

### 6.1 Re-run summary (10 vs 10, fixed epsilon)

- **td_only mean:** -18.70 (std 59.10).
- **full_rmer mean:** -33.72 (std 48.46).
- **Difference:** td_only is **+15** points better on average; **not** statistically significant (p ≈ 0.54).

So with a **correct, consistent** epsilon schedule:
- td_only (Hindsight TD only) is slightly better on average than full RMER (TD + LFIW + TCE).
- full_rmer has **lower variance** (48 vs 59).

### 6.2 What that suggests (no bug, just algorithm / env)

- The **bug** was exploration schedule inconsistency and misinterpreted config. That is fixed.
- Remaining differences are **algorithm and environment**:  
  - LunarLander is high-variance; 10 seeds are not enough to get small p-values.  
  - Adding LFIW + TCE (full_rmer) may stabilize (lower std) but does not improve mean in this run; td_only (prioritization by Hindsight TD only) happens to have a better mean here.

So:
- **“What made it work better”:** Fixing epsilon so that (1) it actually reaches 0.05, and (2) the schedule is in env steps and consistent across seeds. That makes the comparison **valid** and removes the artifact of “half the seeds still exploring.”
- **“Where was the exact problem”:** In `phase1_baseline/dqn.py` (epsilon driven by `train_step` and `update_epsilon()`) and in the trainers (never passing `step` for epsilon). The fix is `decay_epsilon_by_env_step(env_step)` in DQN and the one-line call in each trainer loop.

---

## 7. One-Page Summary

| What | Where | Why it was wrong |
|------|--------|------------------|
| Epsilon decay used **update count** | `dqn.py` `update_epsilon()` called from `update()` | Config is in env steps; code used update steps → 200k env steps = 50k updates ⇒ epsilon stuck at 0.53 for 100k “steps” (updates). |
| Trainer never passed **env step** to agent | `train_rmer.py`, `train_baseline.py` | Agent had no env-step clock; could only use its own update count. |
| Same config used as “steps” in two different senses | `config.py` vs `dqn.py` | Human: “100k steps” = env steps. Code: “100k steps” = updates. Semantic mismatch. |

**Fix:**
- DQN: `decay_epsilon_by_env_step(env_step)` and **do not** call `update_epsilon()` from `update()`.
- Trainers: each step call `decay_epsilon_by_env_step(step)` before `select_action`.
- Config: `epsilon_decay_steps: 20_000` (env steps, 10% of 200k).

**Result:** Epsilon reaches 0.05 by step 20k for every run; seeds 0–9 are comparable; the remaining performance difference between td_only and full_rmer is algorithm/env variance, not an epsilon bug.
