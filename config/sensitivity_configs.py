# sensitivity sweeps (per alpha/beta, temp, lfiw temp)

from config import BASELINE_CONFIG, RMER_CONFIG

BASE_PER = {**BASELINE_CONFIG, "per_alpha": 0.6, "per_beta_start": 0.4}
BASE_RMER = {**RMER_CONFIG}

PER_ALPHA_SWEEP = {
    "per_alpha_0.4": {**BASE_PER, "per_alpha": 0.4},
    "per_alpha_0.6": {**BASE_PER, "per_alpha": 0.6},
    "per_alpha_0.8": {**BASE_PER, "per_alpha": 0.8},
}

PER_BETA_SWEEP = {
    "per_beta_0.3": {**BASE_PER, "per_beta_start": 0.3},
    "per_beta_0.4": {**BASE_PER, "per_beta_start": 0.4},
    "per_beta_0.6": {**BASE_PER, "per_beta_start": 0.6},
}

TEMP_SWEEP = {
    "temp_0.5": {**BASE_RMER, "priority_temperature": 0.5},
    "temp_1.0": {**BASE_RMER, "priority_temperature": 1.0},
    "temp_2.0": {**BASE_RMER, "priority_temperature": 2.0},
}

LFIW_TEMP_SWEEP = {
    "lfiw_temp_5.0": {**BASE_RMER, "lfiw_temperature": 5.0},
    "lfiw_temp_10.0": {**BASE_RMER, "lfiw_temperature": 10.0},
    "lfiw_temp_15.0": {**BASE_RMER, "lfiw_temperature": 15.0},
}

ALL_SWEEPS = {
    **PER_ALPHA_SWEEP,
    **PER_BETA_SWEEP,
    **TEMP_SWEEP,
    **LFIW_TEMP_SWEEP,
}


def get_sensitivity_config(name):
    if name not in ALL_SWEEPS:
        raise KeyError(f"Unknown sensitivity config: {name}. Known: {list(ALL_SWEEPS)}")
    return ALL_SWEEPS[name].copy()
