# single sensitivity run (per_alpha_0.4, temp_1.0, etc). dispatches to PER or RMER

import argparse
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
# print("root", _root)  # debug path

from config.sensitivity_configs import get_sensitivity_config, ALL_SWEEPS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=list(ALL_SWEEPS))
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config_name = args.config
    config = get_sensitivity_config(config_name)
    config["total_steps"] = args.steps

    if config_name.startswith("per_"):
        from phase1_baseline.train_per import PERTrainer
        trainer = PERTrainer(
            env_name=args.env,
            config=config,
            seed=args.seed,
            device=args.device,
            run_suffix=config_name,
        )
        trainer.train(total_steps=args.steps)
        final = trainer.evaluate(num_episodes=10)
        print("\n%s complete! Final eval: %.2f" % (config_name, final))
    else:
        from phase3_rmer.train_rmer import RMERTrainer
        trainer = RMERTrainer(
            env_name=args.env,
            config=config,
            seed=args.seed,
            device=args.device,
            run_suffix=config_name,
        )
        trainer.train(total_steps=args.steps)
        final = trainer.evaluate(num_episodes=10)
        print("\n%s complete! Final eval: %.2f" % (config_name, final))


if __name__ == "__main__":
    main()
