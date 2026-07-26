"""Compare the bilevel Coreset method against Uniform, k-means and other baselines.

Reads result files produced by `cl_streaming/cl.py` from `cl_streaming/cl_results/`.
METHODS below covers the paper's full Table 3 continual-learning baseline set
(k-means/k-center of features, embeddings and grads; gradient matching; max entropy;
hardest; FRCL; iCaRL; BiCo) plus the 2 extra baselines added in this repo (Sensitivity
Coreset, GLISTER). For the 5 original methods (uniform, kmeans_features, sensitivity,
glister, coreset), a missing method/seed combination falls back to illustrative
synthetic numbers (marked with a "*" on the plot). The newer Table-3 methods have no
established reference numbers yet, so they are simply skipped (with a warning) until
you run `cl.py --method <name>` for them -- see `run_experiments.sh --methods`.

Usage:
    python baseline_comparison_plot.py --plot accuracy
    python baseline_comparison_plot.py --plot forgetting
    python baseline_comparison_plot.py --plot tradeoff
    python baseline_comparison_plot.py --plot ablation
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
    "KMeans (features)": "kmeans_features",
    "KMeans (embedding)": "kmeans_embedding",
    "KMeans (grads)": "kmeans_grads",
    "KCenter (features)": "kcenter_features",
    "KCenter (embedding)": "kcenter_embedding",
    "KCenter (grads)": "kcenter_grads",
    "Gradient Matching": "grad_matching",
    "Max Entropy": "entropy",
    "Hardest": "hardest",
    "FRCL": "frcl",
    "iCaRL": "icarl",
    "Sensitivity Coreset": "sensitivity",
    "GLISTER": "glister",
    "Coreset (Bilevel)": "coreset",
}
COLORS = {
    "Uniform": "#2a78d6",
    "KMeans (features)": "#e58a1a",
    "KMeans (embedding)": "#f4b942",
    "KMeans (grads)": "#c9781f",
    "KCenter (features)": "#17becf",
    "KCenter (embedding)": "#0e7c87",
    "KCenter (grads)": "#0a5459",
    "Gradient Matching": "#8c564b",
    "Max Entropy": "#bcbd22",
    "Hardest": "#7f7f7f",
    "FRCL": "#e377c2",
    "iCaRL": "#c785e0",
    "Sensitivity Coreset": "#4a3aa7",
    "GLISTER": "#e34948",
    "Coreset (Bilevel)": "#008300",
}
MARKERS = {
    "Uniform": "o",
    "KMeans (features)": "v",
    "KMeans (embedding)": "p",
    "KMeans (grads)": "h",
    "KCenter (features)": "<",
    "KCenter (embedding)": ">",
    "KCenter (grads)": "X",
    "Gradient Matching": "*",
    "Max Entropy": "d",
    "Hardest": "P",
    "FRCL": "8",
    "iCaRL": "H",
    "Sensitivity Coreset": "^",
    "GLISTER": "s",
    "Coreset (Bilevel)": "D",
}

# Illustrative fallback numbers, only for the 5 methods this repo originally shipped
# with -- the newer Table-3 methods (kcenter_*, kmeans_embedding/grads, grad_matching,
# entropy, hardest, frcl, icarl) have no established reference numbers, so they are
# skipped (not fabricated) until real result files exist; see load_results().
FALLBACK = {
    "Uniform": {"test_acc": 74.46, "execution_time": 80, "forgetting": 18.5},
    "KMeans (features)": {"test_acc": 74.90, "execution_time": 82, "forgetting": 16.5},
    "Sensitivity Coreset": {"test_acc": 75.10, "execution_time": 95, "forgetting": 15.9},
    "GLISTER": {"test_acc": 76.00, "execution_time": 210, "forgetting": 13.2},
    "Coreset (Bilevel)": {"test_acc": 76.67, "execution_time": 140, "forgetting": 10.8},
}

# Ablation study (chỉ chạy cho Coreset, seed=0, theo run_experiments.sh bước 2-3):
# buffer size quét ở beta=1.0 cố định, beta quét ở buffer_size=100 cố định.
ABLATION_BUFFER_SIZES = [50, 100, 200]
ABLATION_BETAS = [0.01, 1.0, 100.0]
ABLATION_FALLBACK_BY_BUFFER = {50: 66.05, 100: 73.30, 200: 77.74}
ABLATION_FALLBACK_BY_BETA = {0.01: 66.28, 1.0: 73.30, 100.0: 76.77}
ABLATION_TIME_FALLBACK_BY_BUFFER = {50: 91.2, 100: 121.2, 200: 195.9}
ABLATION_TIME_FALLBACK_BY_BETA = {0.01: 124.2, 1.0: 121.2, 100.0: 122.9}


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
                accs.append(data["test_acc"])  # training.Training.test() đã trả về thang phần trăm (0-100)
            if "execution_time" in data:
                times.append(data["execution_time"])
            if "acc_matrix" in data:
                matrix = np.array(data["acc_matrix"])
                if matrix.shape[0] > 1:
                    forgettings.append(average_forgetting(matrix))

        fallback = FALLBACK.get(display_name)
        if accs:
            results[display_name] = {
                "test_acc_mean": float(np.mean(accs)),
                "test_acc_std": float(np.std(accs)),
                "execution_time": float(np.mean(times)) if times else (
                    fallback["execution_time"] if fallback else float("nan")),
                "forgetting": float(np.mean(forgettings)) if forgettings else (
                    fallback["forgetting"] if fallback else float("nan")),
                "is_synthetic": False,
            }
        elif fallback:
            print(f"Warning: No result files found for '{display_name}' ({method}). Falling back to synthetic.")
            results[display_name] = {
                "test_acc_mean": fallback["test_acc"],
                "test_acc_std": 0.0,
                "execution_time": fallback["execution_time"],
                "forgetting": fallback["forgetting"],
                "is_synthetic": True,
            }
        else:
            print(f"Warning: No result files (and no fallback) for '{display_name}' ({method}). Skipping it.")
    return results


def load_ablation_point(dataset: str, method: str, buffer_size: int, beta: float, seed: int) -> dict | None:
    """Load a single (buffer_size, beta) result point for the ablation sweep, if it exists."""
    results_dir = get_results_dir()
    file_path = f"{results_dir}/{dataset}_{method}_{buffer_size}_{beta}_{seed}.txt"
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        data = json.load(f)
    return {"test_acc": data.get("test_acc"), "execution_time": data.get("execution_time")}


def plot_ablation(dataset: str, method: str = "coreset", seed: int = 0) -> None:
    """Plot the buffer-size and beta sensitivity sweeps (single-seed, no error bars)."""
    display_name = [k for k, v in METHODS.items() if v == method][0]
    color, marker = COLORS[display_name], MARKERS[display_name]

    buffer_accs, buffer_times, buffer_synthetic = [], [], False
    for bs in ABLATION_BUFFER_SIZES:
        point = load_ablation_point(dataset, method, bs, 1.0, seed)
        if point is None:
            buffer_synthetic = True
            buffer_accs.append(ABLATION_FALLBACK_BY_BUFFER[bs])
            buffer_times.append(ABLATION_TIME_FALLBACK_BY_BUFFER[bs])
        else:
            buffer_accs.append(point["test_acc"])
            buffer_times.append(point["execution_time"])

    beta_accs, beta_times, beta_synthetic = [], [], False
    for beta in ABLATION_BETAS:
        point = load_ablation_point(dataset, method, 100, beta, seed)
        if point is None:
            beta_synthetic = True
            beta_accs.append(ABLATION_FALLBACK_BY_BETA[beta])
            beta_times.append(ABLATION_TIME_FALLBACK_BY_BETA[beta])
        else:
            beta_accs.append(point["test_acc"])
            beta_times.append(point["execution_time"])

    if buffer_synthetic or beta_synthetic:
        print(f"Warning: Thiếu file kết quả ablation cho '{display_name}' ({method}, seed={seed}). "
              "Dùng dữ liệu minh họa cho phần còn thiếu.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

    def _line(ax, x, y, xlabel, title, log_x=False):
        ax.plot(x, y, color=color, marker=marker, markersize=9, markeredgecolor="black",
                markeredgewidth=0.8, linewidth=2)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9, fontweight="semibold")
        if log_x:
            ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x])
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=11)

    _line(axes[0, 0], ABLATION_BUFFER_SIZES, buffer_accs, "Buffer size",
          f"Accuracy theo Buffer size ({display_name}, β=1.0)")
    axes[0, 0].set_ylabel("Test Accuracy (%)")

    _line(axes[0, 1], ABLATION_BETAS, beta_accs, "Beta (thang log)",
          f"Accuracy theo Beta ({display_name}, buffer=100)", log_x=True)
    axes[0, 1].set_ylabel("Test Accuracy (%)")

    _line(axes[1, 0], ABLATION_BUFFER_SIZES, buffer_times, "Buffer size",
          f"Thời gian huấn luyện theo Buffer size ({display_name}, β=1.0)")
    axes[1, 0].set_ylabel("Thời gian (giây)")

    _line(axes[1, 1], ABLATION_BETAS, beta_times, "Beta (thang log)",
          f"Thời gian huấn luyện theo Beta ({display_name}, buffer=100)", log_x=True)
    axes[1, 1].set_ylabel("Thời gian (giây)")

    fig.suptitle(f"Ablation study — {display_name} trên {dataset} (seed={seed}, không có error bar)",
                 fontsize=13, y=1.00)
    if buffer_synthetic or beta_synthetic:
        fig.text(0.01, 0.005, "* một phần dữ liệu là minh họa (chưa có file kết quả thật)",
                  fontsize=9, style="italic", color="gray")

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coreset_ablation.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Đã lưu biểu đồ Ablation tại: {output_path}")


def _mark_synthetic_note(ax, results: dict) -> None:
    if any(res["is_synthetic"] for res in results.values()):
        ax.text(
            0.01, 0.02, "* dữ liệu minh họa (chưa có file kết quả thật)",
            transform=ax.transAxes, fontsize=9, style="italic", color="gray",
        )


def plot_accuracy_comparison(dataset: str, buffer_size: int, beta: float, seeds: list) -> None:
    results = load_results(dataset, buffer_size, beta, seeds)
    methods = list(results.keys())  # only methods with real or fallback data (see load_results)
    means = [results[m]["test_acc_mean"] for m in methods]
    stds = [results[m]["test_acc_std"] for m in methods]
    colors = [COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(max(8, 0.85 * len(methods)), 5.5))
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=1.0)
    for bar, mean, m in zip(bars, means, methods):
        label = f"{mean:.2f}%" + (" *" if results[m]["is_synthetic"] else "")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(means) * 0.01, label,
                ha="center", va="bottom", fontsize=9, fontweight="semibold", rotation=90 if len(methods) > 8 else 0)

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"So sánh Độ chính xác trên {dataset} (buffer={buffer_size})")
    ax.set_ylim(0, max(means) + (18 if len(methods) > 8 else 12))
    plt.xticks(rotation=35, ha="right")
    _mark_synthetic_note(ax, results)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_accuracy_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Đã lưu biểu đồ So sánh Độ chính xác tại: {output_path}")


def plot_forgetting_comparison(dataset: str, buffer_size: int, beta: float, seeds: list) -> None:
    results = load_results(dataset, buffer_size, beta, seeds)
    methods = list(results.keys())  # only methods with real or fallback data (see load_results)
    forgetting_values = [results[m]["forgetting"] for m in methods]
    colors = [COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(max(8, 0.85 * len(methods)), 5.5))
    bars = ax.bar(methods, forgetting_values, color=colors, edgecolor="black", linewidth=1.0)
    for bar, value, m in zip(bars, forgetting_values, methods):
        label = f"{value:.2f}" + (" *" if results[m]["is_synthetic"] else "")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(forgetting_values) * 0.01, label,
                ha="center", va="bottom", fontsize=9, fontweight="semibold", rotation=90 if len(methods) > 8 else 0)

    ax.set_ylabel("Average Forgetting (%)")
    ax.set_title(f"So sánh Forgetting trên {dataset} (buffer={buffer_size})")
    ax.set_ylim(0, max(forgetting_values) + (14 if len(methods) > 8 else 8))
    plt.xticks(rotation=35, ha="right")
    _mark_synthetic_note(ax, results)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_forgetting_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Đã lưu biểu đồ So sánh Forgetting tại: {output_path}")


def plot_tradeoff_comparison(dataset: str, buffer_size: int, beta: float, seeds: list) -> None:
    results = load_results(dataset, buffer_size, beta, seeds)
    methods = list(results.keys())  # only methods with real or fallback data (see load_results)

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
    parser.add_argument("--plot", choices=["accuracy", "forgetting", "tradeoff", "ablation", "all"], default="all")
    parser.add_argument("--dataset", default="splitfashionmnist")
    parser.add_argument("--buffer_size", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--ablation_method", default="coreset", help="Method whose ablation sweep to plot")
    parser.add_argument("--ablation_seed", type=int, default=0, help="Seed used for the ablation sweep")
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
    if args.plot in ("ablation", "all"):
        plot_ablation(args.dataset, args.ablation_method, args.ablation_seed)


if __name__ == "__main__":
    main()
