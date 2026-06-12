"""Plot val/test_score/nq from a Search-R1 training log.

Usage:
    python plot_val_nq.py train.log
    python plot_val_nq.py train.log --metric val/test_score/nq --out nq_score.png
    python plot_val_nq.py train.log --step 0 600
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_metric(filepath: str, metric: str, step_range=None):
    records = []
    last_step = None

    step_pattern = re.compile(r"(?:^|\b)(?:epoch\s+\d+,\s*)?step[:\s]+(\d+)\b")
    metric_pattern = re.compile(
        rf"{re.escape(metric)}\s*[:=]\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)",
        re.IGNORECASE,
    )

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = strip_ansi(raw_line.strip())

            step_match = step_pattern.search(line)
            current_step = int(step_match.group(1)) if step_match else last_step
            if step_match:
                last_step = current_step

            metric_match = metric_pattern.search(line)
            if not metric_match:
                continue

            if current_step is None:
                # Validation before training may be logged before a step appears.
                current_step = 0

            if step_range and not (step_range[0] <= current_step <= step_range[1]):
                continue

            records.append((current_step, float(metric_match.group(1))))

    return records


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return None
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_metric(records, metric: str, out_path: str, ma_window: int):
    if not records:
        raise ValueError(f"No records found for metric: {metric}")

    steps = np.array([x[0] for x in records], dtype=int)
    values = np.array([x[1] for x in records], dtype=float)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, values, marker="o", linewidth=1.0, markersize=3, color="tab:blue", alpha=0.75)

    smoothed = moving_average(values, ma_window)
    if smoothed is not None:
        ax.plot(
            steps[ma_window - 1 :],
            smoothed,
            linewidth=2.0,
            color="tab:red",
            label=f"MA-{ma_window}",
        )
        ax.legend()

    ax.set_title(metric)
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot a validation metric from a Search-R1 log.")
    parser.add_argument("logfile", help="Path to training log")
    parser.add_argument("--metric", default="val/test_score/nq", help="Metric key to parse")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--step", type=int, nargs=2, metavar=("START", "END"), help="Only plot this step range")
    parser.add_argument("--ma", type=int, default=3, help="Moving-average window; use 1 to disable")
    args = parser.parse_args()

    if args.out is None:
        logdir = os.path.dirname(os.path.abspath(args.logfile)) or "."
        safe_metric = args.metric.replace("/", "_")
        args.out = os.path.join(logdir, "plots", f"{safe_metric}.png")

    step_range = tuple(args.step) if args.step else None
    records = parse_metric(args.logfile, args.metric, step_range=step_range)
    print(f"Parsed {len(records)} records for {args.metric}")

    if records:
        print(f"First: step={records[0][0]}, value={records[0][1]:.6f}")
        print(f"Last:  step={records[-1][0]}, value={records[-1][1]:.6f}")

    plot_metric(records, args.metric, args.out, args.ma)
    print(f"[OK] {args.out}")


if __name__ == "__main__":
    main()
