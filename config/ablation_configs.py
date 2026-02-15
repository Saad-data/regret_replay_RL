# ablation configs for rmer component study

from config import BASELINE_CONFIG

BASE = {
    **BASELINE_CONFIG,
    "total_steps": 200000,
}

ABLATION_CONFIGS = {
    "lfiw_only": {
        **BASE,
        "use_hindsight_td": False,
        "use_lfiw": True,
        "use_tce": False,
    },
    "tce_only": {
        **BASE,
        "use_hindsight_td": False,
        "use_lfiw": False,
        "use_tce": True,
    },
    "htd_lfiw": {
        **BASE,
        "use_hindsight_td": True,
        "use_lfiw": True,
        "use_tce": False,
    },
    "htd_tce": {
        **BASE,
        "use_hindsight_td": True,
        "use_lfiw": False,
        "use_tce": True,
    },
    "lfiw_tce": {
        **BASE,
        "use_hindsight_td": False,
        "use_lfiw": True,
        "use_tce": True,
    },
}


def get_config(name):
    return ABLATION_CONFIGS[name].copy()


def get_run_name(config_name, env, seed):
    # tb / run dir name
    env_safe = env.replace("/", "_")
    return f"ablation_{config_name}_{env_safe}_seed{seed}"
