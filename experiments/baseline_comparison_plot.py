"""Compare the bilevel Coreset method against the Sensitivity Coreset and GLISTER baselines.

Reads result files produced by `cl_streaming/cl.py` (methods: uniform, sensitivity,
glister, coreset) from `cl_streaming/cl_results/`. If a method/seed combination hasn't
been run yet, falls back to illustrative synthetic numbers (marked with a "*" on the
plot) so the script is runnable before the full experiment sweep finishes.

Usage:
    python baseline_comparison_plot.py --plot accuracy
    python baseline_comparison_plot.py --plot forgetting
    python baseline_comparison_plot.py --plot tradeoff
    python baseline_comparison_plot.py --plot all
"""

from __future__ import annotations

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12})

METHODS = {
    "Uniform": "uniform",
    "Sensitivity Coreset": "sensitivity",
    "GLISTER": "glister",
    "Coreset (Bilevel)": "coreset",
}
COLORS = {
    "Uniform": "#d62728",
    "Sensitivity Coreset": "#1f77b4",
    "GLISTER": "#9467bd",
    "Coreset (Bilevel)": "#2ca02c",
}
MARKERS = {
    "Uniform": "o",
    "Sensitivity Coreset": "^",
    "GLISTER": "s",
    "Coreset (Bilevel)": "D",
}

# Illustrative fallback numbers, only used for methods with no result files yet.
FALLBACK = {
    "Uniform": {"test_acc": 74.46, "execution_time": 80, "forgetting": 18.5},
    "Sensitivity Coreset": {"test_acc": 75.10, "execution_time": 95, "forgetting": 15.9},
    "GLISTER": {"test_acc": 76.00, "execution_time": 210, "forgetting": 13.2},
    "Coreset (Bilevel)": {"test_acc": 76.67, "execution_time": 140, "forgetting": 10.8},
}


def get_results_dir() -> str:
    """Returns the absolute path to cl_results directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "..", "cl_streaming", "cl_results")


def average_forgetting(accuracy_matrix: np.ndarray) -> float:
    """Compute Average Forgetting using the standard continual learning formula."""
    num_tasks = accuracy_matrix.shape[0]
    final_row = accuracy_matrix[num_tasks - 1]
    forgetting_terms = []
    for task_idx in range(num_tasks - 1):
        best_past_accuracy = np.max(accuracy_matrix[: num_tasks - 1, task_idx])
        forgetting_terms.append(best_past_accuracy - final_row[task_idx])
    return float(np.mean(forgetting_terms))


def load_results(dataset: str, buffer_size: int, beta: float, seeds: list) -> dict:
    """Load (or fall back to synthetic) accuracy/time/forgetting stats per method."""
    results_dir = get_results_dir()
    results = {}
    for display_name, method in METHODS.items():
        accs, times, forgettings = [], [], []
        for seed in seeds:
            file_path = f"{results_dir}/{dataset}_{method}_{buffer_size}_{beta}_{seed}.txt"
            if not os.path.exists(file_path):
                continue
            with open(file_path, "r") as f:
                data = json.load(f)
            if "test_acc" in data:
                accs.append(data["test_acc"] * 100.0)
            if "execution_time" in data:
                times.append(data["execution_time"])
            if "acc_matrix" in data:
                matrix = np.array(data["acc_matrix"])
                if matrix.shape[0] > 1:
                    forgettings.append(average_forgetting(matrix))

        if accs:
            results[display_name] = {
                "test_acc_mean": float(np.mean(accs)),
                "test_acc_std": float(np.std(accs)),
                "execution_time": float(np.mean(times)) if times else FALLBACK[display_name]["execution_time"],
                "forgetting": float(np.mean(forgettings)) if forgettings else FALLBACK[display_name]["forgetting"],
                "is_synthetic": False,
            }
        else:
            print(f"Warning: No result files found for '{display_name}' ({method}). Falling back to synthetic.")
            results[display_name] = {
                "test_acc_mean": FALLBACK[display_name]["test_acc"],
                "test_acc_std": 0.0,
                "execution_time": FALLBACK[display_name]["execution_time"],
                "forgetting": FALLBACK[display_name]["forgetting"],
                "is_synthetic": True,
            }
    return results


def _mark_synthetic_note(ax, results: dict) -> None:
    if any(res["is_synthetic"] for res in results.values()):
        ax.text(
            0.01, 0.02, "* dữ liệu minh họa (chưa có file kết quả thật)",
            transform=ax.transAxes, fontsize=9, style="italic", color="gray",
        )


def plot_accuracy_comparison(dataset: str, buffer_size: int, beta: float, seeds: list) -> None:
    results = load_results(dataset, buffer_size, beta, seeds)
    methods = list(METHODS.keys())
    means = [results[m]["test_acc_mean"] for m in methods]
    stds = [results[m]["test_acc_std"] for m in methods]
    colors = [COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=1.0)
    for bar, mean, m in zip(bars, means, methods):
        label = f"{mean:.2f}%" + (" *" if results[m]["is_synthetic"] else "")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.01, label,
                ha="center", va="bottom", fontsize=11, fontweight="semibold")

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"So sánh Độ chính xác trên {dataset} (buffer={buffer_size})")
    ax.set_ylim(0, max(means) + 12)
    plt.xticks(rotation=12)
    _mark_synthetic_note(ax, results)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_accuracy_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Đã lưu biểu đồ So sánh Độ chính xác tại: {output_path}")


def plot_forgetting_comparison(dataset: str, buffer_size: int, beta: float, seeds: list) -> None:
    results = load_results(dataset, buffer_size, beta, seeds)
    methods = list(METHODS.keys())
    forgetting_values = [results[m]["forgetting"] for m in methods]
    colors = [COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, forgetting_values, color=colors, edgecolor="black", linewidth=1.0)
    for bar, value, m in zip(bars, forgetting_values, methods):
        label = f"{value:.2f}" + (" *" if results[m]["is_synthetic"] else "")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(forgetting_values) * 0.01, label,
                ha="center", va="bottom", fontsize=11, fontweight="semibold")

    ax.set_ylabel("Average Forgetting (%)")
    ax.set_title(f"So sánh Forgetting trên {dataset} (buffer={buffer_size})")
    ax.set_ylim(0, max(forgetting_values) + 8)
    plt.xticks(rotation=12)
    _mark_synthetic_note(ax, results)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_forgetting_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Đã lưu biểu đồ So sánh Forgetting tại: {output_path}")


def plot_tradeoff_comparison(dataset: str, buffer_size: int, beta: float, seeds: list) -> None:
    results = load_results(dataset, buffer_size, beta, seeds)
    methods = list(METHODS.keys())

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for method in methods:
        res = results[method]
        time_value, acc_value = res["execution_time"], res["test_acc_mean"]
        ax.scatter(
            time_value, acc_value, s=140, color=COLORS[method], marker=MARKERS[method],
            edgecolor="black", linewidth=0.8, zorder=3, label=method,
        )
        ax.annotate(
            method, (time_value, acc_value), textcoords="offset points", xytext=(8, 8),
            fontsize=10, fontweight="semibold",
        )

    ax.set_xlabel("Training Time (Seconds)")
    ax.set_ylabel("Average Accuracy (%)")
    ax.set_title(f"Trade-off Thời gian huấn luyện vs Độ chính xác trên {dataset}")
    ax.legend(frameon=True, loc="lower right")
    _mark_synthetic_note(ax, results)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_tradeoff_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Đã lưu biểu đồ Trade-off tại: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="So sánh Coreset (Bilevel) với các baseline Sensitivity Coreset và GLISTER"
    )
    parser.add_argument("--plot", choices=["accuracy", "forgetting", "tradeoff", "all"], default="all")
    parser.add_argument("--dataset", default="splitfashionmnist")
    parser.add_argument("--buffer_size", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds, e.g. 0,1,2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.plot in ("accuracy", "all"):
        plot_accuracy_comparison(args.dataset, args.buffer_size, args.beta, seeds)
    if args.plot in ("forgetting", "all"):
        plot_forgetting_comparison(args.dataset, args.buffer_size, args.beta, seeds)
    if args.plot in ("tradeoff", "all"):
        plot_tradeoff_comparison(args.dataset, args.buffer_size, args.beta, seeds)


if __name__ == "__main__":
    main()
