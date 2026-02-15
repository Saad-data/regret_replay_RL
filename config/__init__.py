# config for baseline dqn + rmer (shared by everyone)
# env targets used by comparison scripts

ENV_CONFIG = {
    'lunar_lander': {'target_reward': 200},
    'cartpole': {'target_reward': 475},
    'mountaincar': {'target_reward': -110},
}

# baseline dqn - used by baseline trainer and by dqn inside rmer
BASELINE_CONFIG = {
    'buffer_size': 100_000,
    'batch_size': 64,
    'total_steps': 200_000,
    'learning_starts': 1000,
    'train_freq': 4,
    'log_freq': 1000,
    'eval_freq': 20000,
    'eval_episodes': 10,
    'save_freq': 50000,
    'hidden_dims': [128, 128],
    'activation': 'relu',
    'learning_rate': 2.5e-4,
    'gamma': 0.99,
    'tau': 0.005,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay_steps': 20_000,
}

PER_CONFIG = {
    'per_alpha': 0.6,
    'per_beta_start': 0.4,
}

HINDSIGHT_TD_CONFIG = {
    'clip_td_error': 10.0,
}

LFIW_CONFIG = {
    'classifier_hidden_dim': 64,
    'classifier_lr': 1e-3,
    'temperature': 10.0,
    'fast_buffer_ratio': 0.2,
    'classifier_update_freq': 4,
    'min_samples_for_training': 2000,
}

TCE_CONFIG = {
    'gamma': 0.99,
    'c_max_suboptimality': 0.1,
    'b1_start': 0.4,
    'b1_end': 0.9,
    'b2_start': 1.6,
    'b2_end': 1.1,
    'moving_avg_window': 1000,
    'moving_avg_momentum': 0.99,
}

RMER_CONFIG = {
    **BASELINE_CONFIG,
    'use_hindsight_td': True,
    'use_lfiw': True,
    'use_tce': True,
    'priority_epsilon': 1e-6,
    'priority_clip_max': 5.0,
    'priority_temperature': 1.0,
}

ABLATION_CONFIGS = {
    'uniform': {'use_hindsight_td': False, 'use_lfiw': False, 'use_tce': False},
    'td_only': {'use_hindsight_td': True, 'use_lfiw': False, 'use_tce': False},
    'lfiw_only': {'use_hindsight_td': False, 'use_lfiw': True, 'use_tce': False},
    'tce_only': {'use_hindsight_td': False, 'use_lfiw': False, 'use_tce': True},
    'td_lfiw': {'use_hindsight_td': True, 'use_lfiw': True, 'use_tce': False},
    'td_tce': {'use_hindsight_td': True, 'use_lfiw': False, 'use_tce': True},
    'lfiw_tce': {'use_hindsight_td': False, 'use_lfiw': True, 'use_tce': True},
    'full_rmer': {'use_hindsight_td': True, 'use_lfiw': True, 'use_tce': True},
}
