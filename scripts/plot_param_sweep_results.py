#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASET_ORDER = ["Cora", "CiteSeer", "PubMed", "Reddit", "ogbn-arxiv", "ogbn-products"]
EXPERIMENT_ORDER = ["sample_size", "init_batch", "samp_dist"]
EXPERIMENT_LABELS = {
    "sample_size": "Sample Size",
    "init_batch": "Init Batch",
    "samp_dist": "Sampling Distribution",
}
ACC_RE = re.compile(r"\[ACC\].*accuracy: ([0-9.]+) %")
BATCH_TIME_RE = re.compile(r"\[BATCH TIME\] ([0-9.]+) seconds")
TOTAL_TIME_RE = re.compile(r"\[TOTAL TIME\] ([0-9.]+) seconds")


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def resolve_log_path(csv_path: Path, raw_log_path: str):
    if not raw_log_path:
        return None

    direct = Path(raw_log_path)
    if direct.exists():
        return direct

    fallback = csv_path.parent / "logs" / direct.name
    if fallback.exists():
        return fallback

    return None


def recover_metrics_from_log(csv_path: Path, row):
    log_path = resolve_log_path(csv_path, row.get("log_path", ""))
    if log_path is None:
        return row

    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return row

    acc_matches = ACC_RE.findall(text)
    batch_matches = BATCH_TIME_RE.findall(text)
    total_matches = TOTAL_TIME_RE.findall(text)
    if acc_matches and batch_matches and total_matches:
        row["status"] = "ok"
        row["acc"] = float(acc_matches[-1])
        row["batch_time"] = float(batch_matches[-1])
        row["total_time"] = float(total_matches[-1])

    return row


def read_records(csv_path: Path):
    records = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["acc"] = parse_float(row.get("acc", ""))
            row["batch_time"] = parse_float(row.get("batch_time", ""))
            row["total_time"] = parse_float(row.get("total_time", ""))
            if row.get("status") != "ok" or not all(
                math.isfinite(row[key]) for key in ["acc", "batch_time", "total_time"]
            ):
                row = recover_metrics_from_log(csv_path, row)
            records.append(row)
    return records


def sort_key_for_value(experiment: str, value: str):
    if experiment in {"sample_size", "init_batch"}:
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)
    if experiment == "samp_dist":
        order = {"uniform": 0, "importance": 1}
        return (order.get(value, 99), value)
    return (0, value)


def ordered_datasets(records):
    present = {row["dataset"] for row in records}
    ordered = [d for d in DATASET_ORDER if d in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def ordered_experiments(records):
    present = {row["experiment"] for row in records}
    ordered = [e for e in EXPERIMENT_ORDER if e in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def build_summary(records):
    grouped = defaultdict(lambda: {"acc": [], "batch_time": [], "total_time": []})
    statuses = defaultdict(lambda: {"ok": 0, "failed": 0, "oom": 0})

    for row in records:
        key = (row["dataset"], row["experiment"], row["param_value"])
        status = row["status"]
        if status == "ok":
            statuses[key]["ok"] += 1
            for metric in ["acc", "batch_time", "total_time"]:
                if math.isfinite(row[metric]):
                    grouped[key][metric].append(row[metric])
        elif status == "oom":
            statuses[key]["oom"] += 1
        else:
            statuses[key]["failed"] += 1

    summary = {}
    keys = sorted(grouped.keys() | statuses.keys(), key=lambda item: (item[0], item[1], sort_key_for_value(item[1], item[2])))
    for key in keys:
        entry = {
            "dataset": key[0],
            "experiment": key[1],
            "param_value": key[2],
        }
        for metric in ["acc", "batch_time", "total_time"]:
            values = np.array(grouped[key][metric], dtype=float)
            if len(values) > 0:
                entry[f"{metric}_n"] = int(len(values))
                entry[f"{metric}_mean"] = float(np.mean(values))
                entry[f"{metric}_std"] = float(np.std(values, ddof=0))
            else:
                entry[f"{metric}_n"] = 0
                entry[f"{metric}_mean"] = math.nan
                entry[f"{metric}_std"] = math.nan
        entry["ok_runs"] = statuses[key]["ok"]
        entry["failed_runs"] = statuses[key]["failed"]
        entry["oom_runs"] = statuses[key]["oom"]
        summary[key] = entry
    return summary


def write_summary_csv(summary, out_path: Path):
    fieldnames = [
        "dataset",
        "experiment",
        "param_value",
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(summary.keys(), key=lambda item: (item[0], item[1], sort_key_for_value(item[1], item[2]))):
            writer.writerow(summary[key])


def values_for_dataset_experiment(summary, dataset: str, experiment: str):
    keys = [key for key in summary if key[0] == dataset and key[1] == experiment]
    return sorted(keys, key=lambda item: sort_key_for_value(item[1], item[2]))


def plot_dataset_experiment(summary, dataset: str, experiment: str, output_file: Path):
    keys = values_for_dataset_experiment(summary, dataset, experiment)
    if not keys:
        return

    values = [key[2] for key in keys]
    is_numeric = experiment in {"sample_size", "init_batch"}
    if is_numeric:
        xs = np.array([float(value) for value in values], dtype=float)
    else:
        xs = np.arange(len(values))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("acc", "Micro-F1 (%)"),
        ("batch_time", "Batch Time (s)"),
        ("total_time", "Total Time (s)"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        means = np.array([summary[key][f"{metric}_mean"] for key in keys], dtype=float)
        stds = np.nan_to_num(np.array([summary[key][f"{metric}_std"] for key in keys], dtype=float), nan=0.0)

        if is_numeric:
            ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, linewidth=2)
            ax.set_xlabel(EXPERIMENT_LABELS.get(experiment, experiment))
        else:
            ax.bar(xs, means, yerr=stds, capsize=3, alpha=0.9)
            ax.set_xticks(xs)
            ax.set_xticklabels(values)
            ax.set_xlabel(EXPERIMENT_LABELS.get(experiment, experiment))

        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    fig.suptitle(f"{dataset}: {EXPERIMENT_LABELS.get(experiment, experiment)} Sweep")
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot parameter sweep comparisons from results.csv")
    parser.add_argument("--input", required=True, help="Path to results.csv from run_param_sweeps.sh")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    records = read_records(input_path)
    summary = build_summary(records)
    summary_csv = outdir / "summary.csv"
    write_summary_csv(summary, summary_csv)

    for dataset in ordered_datasets(records):
        for experiment in ordered_experiments(records):
            plot_dataset_experiment(
                summary=summary,
                dataset=dataset,
                experiment=experiment,
                output_file=outdir / f"{dataset}_{experiment}.png",
            )

    print(f"Saved summary to: {summary_csv}")
    print(f"Saved plots under: {outdir}")


if __name__ == "__main__":
    main()
