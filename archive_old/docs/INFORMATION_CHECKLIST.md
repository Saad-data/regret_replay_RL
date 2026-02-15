# Information Checklist: Seeds 0–4 vs 5–9

## A. Seeds 0–4 vs 5–9: When were they run?

### File timestamps (full_rmer)

```text
# Seeds 5–9 (full_rmer): evening Feb 8
Feb  8 22:46  rmer_LunarLander-v2_seed9
Feb  8 22:41  rmer_LunarLander-v2_seed8
Feb  8 22:28  rmer_LunarLander-v2_seed7
Feb  8 22:27  rmer_LunarLander-v2_seed6
Feb  8 22:22  rmer_LunarLander-v2_seed5

# Seeds 0–4 (full_rmer): morning Feb 8
Feb  8 08:43  rmer_LunarLander-v2_seed4
Feb  8 08:32  rmer_LunarLander-v2_seed3
Feb  8 08:21  rmer_LunarLander-v2_seed2
Feb  8 08:10  rmer_LunarLander-v2_seed1
Feb  8 07:57  rmer_LunarLander-v2_seed0
```

**Conclusion:** Seeds 0–4 were run **~14 hours before** seeds 5–9 (morning vs evening Feb 8). So they were run at **different times**; a config or code change between the two batches is plausible.

---

## B. Epsilon curve check (from `check_epsilon_curves.py`)

| Seed | Epsilon reaches 0.05 at step | Epsilon at end |
|------|-----------------------------|----------------|
| **Seed 0** (td_only & full_rmer) | **~41,000** | 0.050 |
| **Seed 5** (td_only & full_rmer) | **Never** | 0.530 |

- **Seed 0:** Decay over ~40k **env steps** → consistent with **env-step** decay (e.g. `epsilon_decay_steps` ≈ 40k–50k env steps).
- **Seed 5:** Stuck at 0.53 → consistent with **update-step** decay and `epsilon_decay_steps` = 100k updates (50k updates at 200k env steps ⇒ ε ≈ 0.53).

**TensorBoard:** Run `tensorboard --logdir runs/` and check the **Epsilon** scalar for `rmer_LunarLander-v2_seed0` and `rmer_LunarLander-v2_seed5` (and their `_noLFIW_noTCE` variants) to see the curves.

---

## C. Config history

- **Current repo:** `config.py` has `epsilon_decay_steps: 20_000` (after the fix; was 100_000 earlier in this conversation).
- **No git history was inspected** here, so we cannot say whether it was ever 10,000 or changed mid-experiment.
- **What we know:** Seeds 0–4 behave like env-step decay over ~40k; seeds 5–9 behave like update-step decay with 100k. So either:
  - Different **code** (env-step vs update-step) was used, or  
  - Different **config** (e.g. 40k vs 100k, or env vs update interpretation) was used between the two batches.

---

## D. Answers to your questions

### How did seeds 0–4 reach epsilon = 0.05?

**Different code version or different config.** Evidence: seed 0 reaches 0.05 by ~41k **env** steps; seed 5 never does and ends at 0.53 with **update**-based decay. Timestamps show 0–4 and 5–9 run ~14 hours apart, so a change between batches is plausible. We cannot distinguish “different code” vs “different config” from the repo alone.

### What does TensorBoard show for epsilon curves?

- **Seed 0:** Epsilon reaches 0.05 at step **~41,000**.
- **Seed 5:** Epsilon **never** reaches 0.05 (ends at **0.53**).

### Do you want 100k or 10k epsilon decay?

- **Current config (after fix):** `epsilon_decay_steps = 20_000` (10% of 200k). So decay is **not** 100k anymore; it’s 20k.
- **If you prefer 10k:** Set `epsilon_decay_steps = 10_000` in `config.py` for decay over the first 5% of training.
- **Pros of 20k (current):** Standard “~10% of training” decay; ~90% exploitation.  
- **Pros of 10k:** Even more exploitation; matches “10k decay” in the checklist.

### Which action plan?

Recommendation from the evidence above:

- **Option B: Re-run everything (0–9)**  
  Seeds 0–4 and 5–9 used different decay behavior and were run at different times. For a single, comparable experiment, re-run **all** seeds 0–9 with the **fixed code** (env-step decay + current config, e.g. 20k or 10k) for both td_only and full_rmer. Then you have one consistent comparison.

- **Option D (optional):** Before a full re-run, run **one** seed (e.g. seed 10) and confirm in TensorBoard that Epsilon reaches 0.05 by the intended step (e.g. 20k or 10k), then do Option B.

---

## Summary table

| Item | Answer |
|------|--------|
| Seeds 0–4 run time | Morning Feb 8 (dirs 07:57–08:43) |
| Seeds 5–9 run time | Evening Feb 8 (dirs 21:57–22:46) |
| Seed 0: epsilon → 0.05 at step | **~41,000** |
| Seed 5: epsilon → 0.05 at step | **Never** (ends 0.53) |
| Current decay in config | **20,000** env steps (10% of 200k) |
| Recommended action | **Option B** (re-run all 0–9 with fixed code) |
