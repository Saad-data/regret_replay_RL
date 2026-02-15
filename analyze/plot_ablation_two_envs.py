# bar chart: main methods + ablations for cartpole and mountaincar

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = "results/complete_ALL_results.json"
OUT_PATH = "plots/ablation_two_environments.png"


def load_results():
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"Run collect_ALL_results.py first. Missing: {RESULTS_PATH}"
        )
    with open(RESULTS_PATH) as f:
        return json.load(f)


def plot_env(ax, env_name, main_methods, ablation_methods, title, success_threshold=None):
    labels = []
    means = []
    stds = []
    colors = []

    # Main methods
    main_order = [
        "Uniform",
        "PER",
        "TD-only",
        "Full-RMER (ReMERT)",
        "Full-RMER (ReMERN)",
    ]
    color_main = plt.cm.tab10(np.linspace(0, 0.5, len(main_order)))
    for i, name in enumerate(main_order):
        if name not in main_methods:
            continue
        r = main_methods[name]
        labels.append(name.replace("Full-RMER (ReMERT)", "ReMERT").replace("Full-RMER (ReMERN)", "ReMERN"))
        means.append(r["mean"])
        stds.append(r["std"])
        colors.append(color_main[i])

    if ablation_methods:
        n_main = len(labels)
        color_abl = plt.cm.Set3(np.linspace(0, 0.8, len(ablation_methods)))
        for i, (name, r) in enumerate(ablation_methods.items()):
            labels.append(name)
            means.append(r["mean"])
            stds.append(r["std"])
            colors.append(color_abl[i])

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean final eval reward", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    if success_threshold is not None:
        ax.axhline(y=success_threshold, color="green", linestyle="--", alpha=0.6, label=f"Success ({success_threshold})")
        ax.legend(loc="lower right", fontsize=8)
    lo = min(means) - max(30, np.std(means) or 30)
    hi = max(means) + max(30, np.std(means) or 30)
    if success_threshold is not None and success_threshold > max(means):
        hi = max(hi, success_threshold + 50)
    ax.set_ylim(bottom=lo, top=hi)


def main():
    data = load_results()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # CartPole-v1: main + ablations
    main_cp = data.get("CartPole-v1", {})
    abl_cp = data.get("Ablations_CartPole", {})
    plot_env(
        ax1,
        "CartPole-v1",
        main_cp,
        abl_cp,
        "CartPole-v1: Main methods + Ablations",
        success_threshold=450,
    )

    # MountainCar-v0: main + ablations (if collected)
    main_mc = data.get("MountainCar-v0", {})
    abl_mc = data.get("Ablations_MountainCar", {})
    plot_env(
        ax2,
        "MountainCar-v0",
        main_mc,
        abl_mc,
        "MountainCar-v0: Main methods + Ablations",
        success_threshold=-110,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
