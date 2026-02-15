# ReMERN Training and Test Results

**Method:** ReMERN (RMER with Error Network instead of TCE)  
**Environments:** CartPole-v1, MountainCar-v0  
**Seeds:** 0–9 (10 per environment)  
**Training:** 20,000 steps per run (after script update; some earlier runs used 200,000 steps)

---

## CartPole-v1

| Seed | Final eval reward |
|------|-------------------|
| 0    | 480.00            |
| 1    | 213.00            |
| 2    | 500.00            |
| 3    | 500.00            |
| 4    | 500.00            |
| 5    | 500.00            |
| 6    | 500.00            |
| 7    | 500.00            |
| 8    | 156.50            |
| 9    | 500.00            |

- **Mean:** 434.95  
- **Std:** 132.68  
- **Median:** 500.00  
- **Min / Max:** 156.50 / 500.00  
- **n:** 10  

---

## MountainCar-v0

| Seed | Final eval reward |
|------|-------------------|
| 0–9  | -200.00 (all)     |

- **Mean:** -200.00  
- **Std:** 0.00  
- **Median:** -200.00  
- **Min / Max:** -200.00 / -200.00  
- **n:** 10  
- **Note:** MountainCar is difficult; -200 is the usual “no solve” return. Success is often defined as reward > -110.

---

## Quick test (10k steps, seed 99)

- **CartPole-v1:** `python phase3_rmer/train_remern.py --env CartPole-v1 --seed 99 --steps 10000` completes successfully (final eval varies by run).

---

## Files

- **TensorBoard runs:** `runs/remern_CartPole-v1_seed{0-9}`, `runs/remern_MountainCar-v0_seed{0-9}` (and seed 99 if run).
- **Collected JSON:** `results/remern_results.json` (from `python collect_remern_results.py`).
- **Training log:** `remern.log`.
