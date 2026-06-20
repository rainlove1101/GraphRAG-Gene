import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FIG_DPI = 300
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

COLORS = {
    "blue": "#2F6F9F",
    "green": "#4F8A5B",
    "gold": "#C9952E",
    "red": "#A94E4E",
    "gray": "#6B7280",
}


def wrapped_method_label(label):
    replacements = {
        "Gene-only retrieval": "Gene-only\nretrieval",
        "Pathway-only retrieval": "Pathway-only\nretrieval",
        "GraphRAG-Gene full": "GraphRAG-Gene\nfull",
        "No direct-relation bonus": "No direct-relation\nbonus",
        "No specificity weighting": "No specificity\nweighting",
        "Overlap count only": "Overlap count\nonly",
    }
    return replacements.get(label, label)


def save_figure(fig, output_path):
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=FIG_DPI)
    fig.savefig(output_path.with_suffix(".pdf"))


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_metric_bars(df, title, output_path):
    metrics = ["top1_hit_rate", "top3_hit_rate", "top5_hit_rate"]
    labels = ["Top-1", "Top-3", "Top-5"]
    methods = df["method"].tolist()
    x = range(len(methods))
    width = 0.24
    colors = [COLORS["blue"], COLORS["green"], COLORS["gold"]]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        values = df[metric].astype(float).tolist()
        bars = ax.bar([v + offset for v in x], values, width=width, label=labels[i], color=colors[i])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_title(title, pad=10)
    ax.set_ylabel("Hit rate")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(list(x))
    ax.set_xticklabels([wrapped_method_label(m) for m in methods])
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_resolution_sweep(df, output_path):
    fig, ax1 = plt.subplots(figsize=(6.6, 4.1))
    ax1.plot(df["resolution"], df["communities"], marker="o", color=COLORS["blue"], label="Communities")
    ax1.set_xlabel("Leiden resolution")
    ax1.set_ylabel("Communities", color=COLORS["blue"])
    ax1.tick_params(axis="y", labelcolor=COLORS["blue"])
    style_axes(ax1)

    ax2 = ax1.twinx()
    ax2.plot(df["resolution"], df["max_size"], marker="s", color=COLORS["red"], label="Maximum community size")
    ax2.set_ylabel("Maximum community size", color=COLORS["red"])
    ax2.tick_params(axis="y", labelcolor=COLORS["red"])
    ax2.spines["top"].set_visible(False)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2)
    ax1.set_title("Leiden Resolution Sensitivity", pad=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_path)
    plt.close(fig)


def plot_community_distribution(index_dir, output_path):
    nodes_path = Path(index_dir) / "create_final_nodes.parquet"
    nodes = pd.read_parquet(nodes_path)
    sizes = nodes.groupby("community").size()

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.hist(sizes, bins=40, color=COLORS["gray"], edgecolor="white")
    ax.set_yscale("log")
    ax.set_title("Community Size Distribution", pad=10)
    ax.set_xlabel("Community size")
    ax.set_ylabel("Number of communities, log scale")
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables_dir", default="results/tables")
    parser.add_argument("--index_dir", default="results/index")
    parser.add_argument("--output_dir", default="results/figures")
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = tables_dir / "baseline_comparison.csv"
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        plot_metric_bars(
            baseline_df,
            "Baseline Retrieval Performance",
            output_dir / "baseline_retrieval_performance.png",
        )

    ablation_path = tables_dir / "ablation_comparison.csv"
    if ablation_path.exists():
        ablation_df = pd.read_csv(ablation_path)
        plot_metric_bars(
            ablation_df,
            "GraphRAG-Gene Ablation Performance",
            output_dir / "ablation_performance.png",
        )

    resolution_path = tables_dir / "resolution_sweep.csv"
    if resolution_path.exists():
        resolution_df = pd.read_csv(resolution_path)
        plot_resolution_sweep(resolution_df, output_dir / "leiden_resolution_sensitivity.png")

    nodes_path = Path(args.index_dir) / "create_final_nodes.parquet"
    if nodes_path.exists():
        plot_community_distribution(args.index_dir, output_dir / "community_size_distribution.png")

    print(f"Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
