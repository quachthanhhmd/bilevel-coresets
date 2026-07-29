"""Plot continual learning metrics with a single CLI entrypoint.

Usage:
    python tradeoff_time_accuracy_plot.py --plot forgetting
    python tradeoff_time_accuracy_plot.py --plot tradeoff
"""

from __future__ import annotations

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reuse the single source of truth for method names/colors/markers/fallback numbers
# so this script always shows the same methods as baseline_comparison_plot.py.
from baseline_comparison_plot import METHODS, COLORS, MARKERS, FALLBACK, get_results_dir


sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12})


def load_real_accuracy_matrices(results_dir: str = None) -> dict[str, np.ndarray]:
    """Load and average accuracy matrices from real Kaggle runs."""
    if results_dir is None:
        results_dir = get_results_dir()

    dataset = "splitfashionmnist"
    seeds = [0, 1, 2]

    matrices = {}
    for display_name, method in METHODS.items():
        all_matrices = []
        for seed in seeds:
            file_path = f"{results_dir}/{dataset}_{method}_100_1.0_{seed}.txt"
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if "acc_matrix" in data:
                        all_matrices.append(np.array(data["acc_matrix"]))
        
        if all_matrices:
            # Trung bình cộng ma trận theo các seed. training.Training.test() đã trả về
            # giá trị phần trăm (0-100) nên không cần nhân lại với 100 ở đây.
            matrices[display_name] = np.mean(all_matrices, axis=0)
        elif display_name in FALLBACK:
            print(f"Warning: No valid acc_matrix found for {display_name}. Falling back to synthetic.")
            # Dummy fallback, only for the 5 methods this repo originally shipped with.
            matrices[display_name] = np.array([
                [89.0, 0.0, 0.0, 0.0, 0.0],
                [84.0, 87.0, 0.0, 0.0, 0.0],
                [73.0, 76.0, 85.0, 0.0, 0.0],
                [62.0, 64.0, 70.0, 83.0, 0.0],
                [52.0, 54.0, 58.0, 68.0, 81.0],
            ])
        else:
            print(f"Warning: No valid acc_matrix (and no fallback) for {display_name}. Skipping it.")

    return matrices


def average_forgetting(accuracy_matrix: np.ndarray) -> float:
    """Compute Average Forgetting using the standard continual learning formula."""

    num_tasks = accuracy_matrix.shape[0]
    final_row = accuracy_matrix[num_tasks - 1]
    forgetting_terms = []

    for task_idx in range(num_tasks - 1):
        best_past_accuracy = np.max(accuracy_matrix[: num_tasks - 1, task_idx])
        forgetting_terms.append(best_past_accuracy - final_row[task_idx])

    return float(np.mean(forgetting_terms))


def plot_average_forgetting() -> None:
    matrices = load_real_accuracy_matrices()
    methods = list(matrices.keys())
    forgetting_values = [average_forgetting(matrices[method]) for method in methods]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = [COLORS[m] for m in methods]
    bars = ax.bar(methods, forgetting_values, color=colors, edgecolor="black", linewidth=1.0)

    ax.set_ylabel("Average Forgetting (%)")
    ax.set_title("Average Forgetting on SplitFashionMNIST (5 tasks)")
    ax.set_ylim(0, max(forgetting_values) + 10)

    for bar, value in zip(bars, forgetting_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
        )

    plt.tight_layout()
    # plt.show()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "average_forgetting.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Đã lưu biểu đồ Forgetting tại: {output_path}")


def plot_tradeoff() -> None:
    dataset = "splitfashionmnist"
    seeds = [0, 1, 2]
    results_dir = get_results_dir()

    methods = []
    training_time = []
    average_accuracy = []

    for display_name, method in METHODS.items():
        times = []
        accs = []
        for seed in seeds:
            file_path = f"{results_dir}/{dataset}_{method}_100_1.0_{seed}.txt"
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if "execution_time" in data and "test_acc" in data:
                        times.append(data["execution_time"])
                        accs.append(data["test_acc"])  # đã ở thang phần trăm (0-100)

        fallback = FALLBACK.get(display_name)
        if times and accs:
            methods.append(display_name)
            training_time.append(np.mean(times))
            average_accuracy.append(np.mean(accs))
        elif fallback:
            print(f"Warning: Missing real data for {display_name}. Using synthetic fallback.")
            methods.append(display_name)
            training_time.append(fallback["execution_time"])
            average_accuracy.append(fallback["test_acc"])
        else:
            print(f"Warning: Missing real data (and no fallback) for {display_name}. Skipping it.")

    colors = [COLORS[m] for m in methods]
    markers = [MARKERS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    for method, time_value, acc_value, color, marker in zip(
        methods, training_time, average_accuracy, colors, markers
    ):
        ax.scatter(
            time_value,
            acc_value,
            s=130,
            color=color,
            marker=marker,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
            label=method,
        )
        ax.vlines(time_value, ymin=0, ymax=acc_value, colors=color, linestyles="dashed", alpha=0.65)
        ax.hlines(acc_value, xmin=0, xmax=time_value, colors=color, linestyles="dashed", alpha=0.65)
        ax.annotate(
            method,
            (time_value, acc_value),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=11,
            fontweight="semibold",
        )

    ax.set_xlabel("Training Time (Seconds)")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title("Trade-off Between Training Time and Average Accuracy")
    ax.set_xlim(0, max(training_time) * 1.25)
    ax.set_ylim(0, max(average_accuracy) * 1.15)
    ax.legend(frameon=True, loc="lower right")

    plt.tight_layout()
    # plt.show()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tradeoff_time_accuracy.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Đã lưu biểu đồ Trade-off tại: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot continual learning metrics")
    parser.add_argument(
        "--plot",
        choices=["forgetting", "tradeoff"],
        required=True,
        help="Select which plot to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot == "forgetting":
        plot_average_forgetting()
    elif args.plot == "tradeoff":
        plot_tradeoff()
    else:
        raise ValueError(f"Unknown plot type: {args.plot}")


if __name__ == "__main__":
    main()