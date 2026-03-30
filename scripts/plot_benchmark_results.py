#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODE_ORDER = ["fast_full", "gcn_full", "fast_mini"]
MODE_LABELS = {
    "fast_full": "FastGCN (full inf)",
    "gcn_full": "GCN (full inf)",
    "fast_mini": "FastGCN (mini inf)",
}
DATASET_ORDER = ["Cora", "CiteSeer", "PubMed", "Reddit", "ogbn-arxiv", "ogbn-products"]


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def read_records(csv_path: Path):
    records = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["acc"] = parse_float(row.get("acc", ""))
            row["batch_time"] = parse_float(row.get("batch_time", ""))
            row["total_time"] = parse_float(row.get("total_time", ""))
            records.append(row)
    return records


def build_summary(records):
    grouped = defaultdict(lambda: {"acc": [], "batch_time": [], "total_time": []})
    statuses = defaultdict(lambda: {"ok": 0, "failed": 0, "oom": 0})

    for row in records:
        key = (row["dataset"], row["mode"])
        status = row["status"]

        if status == "ok":
            statuses[key]["ok"] += 1
            if math.isfinite(row["acc"]):
                grouped[key]["acc"].append(row["acc"])
            if math.isfinite(row["batch_time"]):
                grouped[key]["batch_time"].append(row["batch_time"])
            if math.isfinite(row["total_time"]):
                grouped[key]["total_time"].append(row["total_time"])
        elif status == "oom":
            statuses[key]["oom"] += 1
        else:
            statuses[key]["failed"] += 1

    summary = {}
    keys = sorted(set(list(grouped.keys()) + list(statuses.keys())))
    for key in keys:
        entry = {}
        for metric in ["acc", "batch_time", "total_time"]:
            values = np.array(grouped[key][metric], dtype=float)
            if len(values) > 0:
                entry[f"{metric}_mean"] = float(np.mean(values))
                entry[f"{metric}_std"] = float(np.std(values, ddof=0))
                entry[f"{metric}_n"] = int(len(values))
            else:
                entry[f"{metric}_mean"] = math.nan
                entry[f"{metric}_std"] = math.nan
                entry[f"{metric}_n"] = 0
        entry["ok_runs"] = statuses[key]["ok"]
        entry["failed_runs"] = statuses[key]["failed"]
        entry["oom_runs"] = statuses[key]["oom"]
        summary[key] = entry
    return summary


def ordered_datasets(summary):
    present = {dataset for dataset, _ in summary.keys()}
    ordered = [d for d in DATASET_ORDER if d in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def ordered_modes(summary):
    present = {mode for _, mode in summary.keys()}
    ordered = [m for m in MODE_ORDER if m in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def write_summary_csv(summary, out_path: Path, datasets, modes):
    fields = [
        "dataset",
        "mode",
        "ok_runs",
        "failed_runs",
        "oom_runs",
        "acc_n",
        "acc_mean",
        "acc_std",
        "batch_time_n",
        "batch_time_mean",
        "batch_time_std",
        "total_time_n",
        "total_time_mean",
        "total_time_std",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dataset in datasets:
            for mode in modes:
                key = (dataset, mode)
                if key not in summary:
                    continue
                row = {"dataset": dataset, "mode": mode}
                row.update(summary[key])
                writer.writerow(row)


def metric_arrays(summary, datasets, modes, metric):
    means = {}
    stds = {}
    for mode in modes:
        mode_means = []
        mode_stds = []
        for dataset in datasets:
            key = (dataset, mode)
            if key in summary:
                mode_means.append(summary[key][f"{metric}_mean"])
                mode_stds.append(summary[key][f"{metric}_std"])
            else:
                mode_means.append(math.nan)
                mode_stds.append(math.nan)
        means[mode] = np.array(mode_means, dtype=float)
        stds[mode] = np.nan_to_num(np.array(mode_stds, dtype=float), nan=0.0)
    return means, stds


def plot_grouped_bars(summary, datasets, modes, metric, ylabel, output_file: Path, log_scale=False):
    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.5), 5))
    if not datasets or not modes:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_file, dpi=180)
        plt.close(fig)
        return

    x = np.arange(len(datasets))
    width = 0.8 / max(len(modes), 1)
    means, stds = metric_arrays(summary, datasets, modes, metric)

    for idx, mode in enumerate(modes):
        pos = x - 0.4 + (idx + 0.5) * width
        ax.bar(pos, means[mode], width=width, label=MODE_LABELS.get(mode, mode), yerr=stds[mode], capsize=3, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Dataset")
    if log_scale:
        positive_values = []
        for mode in modes:
            vals = means[mode]
            positive_values.extend([v for v in vals if math.isfinite(v) and v > 0])
        if positive_values:
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.95, "No positive values for log scale", ha="center", va="top", transform=ax.transAxes, fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def plot_tradeoff(summary, datasets, modes, output_file: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    has_points = False
    for mode in modes:
        xs = []
        ys = []
        labels = []
        for dataset in datasets:
            key = (dataset, mode)
            if key not in summary:
                continue
            x = summary[key]["total_time_mean"]
            y = summary[key]["acc_mean"]
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            xs.append(x)
            ys.append(y)
            labels.append(dataset)

        if not xs:
            continue
        has_points = True
        ax.scatter(xs, ys, s=60, alpha=0.9, label=MODE_LABELS.get(mode, mode))
        for x, y, name in zip(xs, ys, labels):
            ax.annotate(name, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("Total Time (s, log scale)")
    ax.set_ylabel("Micro-F1 (%)")
    ax.set_title("F1-Time Tradeoff")
    ax.grid(alpha=0.25)
    if has_points:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid tradeoff points", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark comparisons from results.csv")
    parser.add_argument("--input", required=True, help="Path to results.csv from run_readme_benchmarks.sh")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    records = read_records(input_path)
    summary = build_summary(records)
    datasets = ordered_datasets(summary)
    modes = ordered_modes(summary)

    summary_csv = outdir / "summary.csv"
    write_summary_csv(summary, summary_csv, datasets, modes)

    plot_grouped_bars(
        summary=summary,
        datasets=datasets,
        modes=modes,
        metric="acc",
        ylabel="Micro-F1 (%)",
        output_file=outdir / "comparison_f1.png",
        log_scale=False,
    )
    plot_grouped_bars(
        summary=summary,
        datasets=datasets,
        modes=modes,
        metric="total_time",
        ylabel="Total Time (s)",
        output_file=outdir / "comparison_total_time.png",
        log_scale=True,
    )
    plot_grouped_bars(
        summary=summary,
        datasets=datasets,
        modes=modes,
        metric="batch_time",
        ylabel="Batch Time (s)",
        output_file=outdir / "comparison_batch_time.png",
        log_scale=True,
    )
    plot_tradeoff(
        summary=summary,
        datasets=datasets,
        modes=modes,
        output_file=outdir / "tradeoff_f1_vs_total_time.png",
    )

    print(f"Saved summary to: {summary_csv}")
    print(f"Saved plots under: {outdir}")


if __name__ == "__main__":
    main()
