"""
解析训练日志，生成训练指标曲线图。

用法:
    python plot_metrics.py nq-search-r1-grpo-qwen2.5-3b-em.log
    python plot_metrics.py nq-search-r1-grpo-qwen2.5-3b-em.log --step 50 300   # 只看 step 50-300
    python plot_metrics.py nq-search-r1-grpo-qwen2.5-3b-em.log --live          # 实时模式，每30秒刷新

输出: 同目录下生成一个 plots/ 文件夹，内含多个 PNG 图片
"""

import argparse
import os
import re
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 指标分组 ──────────────────────────────────────────
PANELS = {
    "Score & Reward": [
        "critic/score/mean", "critic/score/max", "critic/score/min",
        "critic/rewards/mean", "critic/advantages/mean",
    ],
    "Agent Behavior": [
        "env/finish_ratio",
        "env/number_of_actions/mean",
        "env/ratio_of_valid_action",
        "env/number_of_valid_search",
    ],
    "Actor Losses": [
        "actor/pg_loss",
        "actor/kl_loss",
        "actor/entropy_loss",
        "actor/pg_clipfrac",
        "actor/ppo_kl",
    ],
    "Grad & LR": [
        "actor/grad_norm",
        "actor/lr",
        "mfu/actor",
    ],
    "Response & Tokens": [
        "response_length/mean",
        "response_length/max",
        "state_tokens/coverage",
        "prompt_length/mean",
    ],
    "Timing (s)": [
        "timing_s/step",
        "timing_s/gen",
        "timing_s/update_actor",
        "timing_s/ref",
    ],
    "KL & Entropy (detail)": [
        "actor/kl_loss",
        "actor/kl_coef",
        "actor/entropy_loss",
    ],
}


def strip_ansi(text: str) -> str:
    """去掉 ANSI 颜色码和终端控制序列。"""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_log(filepath: str, step_range: tuple = None):
    """从日志文件中解析每步的指标字典。"""
    records = []
    # 匹配行中的 "step:数字 - key:val ..."   (不用 ^ 开头，因为有 ANSI 码 / pid 前缀)
    pattern_line = re.compile(r"step:(\d+)\s+-?\s*(.*?)\s*$")
    # 匹配 "key:val" 对
    pattern_kv = re.compile(r"([\w/]+):(-?[\d.]+(?:e[+-]?\d+)?)")

    total_lines = 0
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            total_lines += 1
            line = strip_ansi(raw_line.strip())
            m = pattern_line.search(line)
            if not m:
                continue
            step = int(m.group(1))
            if step_range and not (step_range[0] <= step <= step_range[1]):
                continue
            tail = m.group(2)
            kv = dict(pattern_kv.findall(tail))
            if kv:
                kv["step"] = step
                records.append(kv)

    if not records:
        # 诊断：统计含 "step:" 的行
        print(f"[DEBUG] Scanned {total_lines} lines, no step lines matched.")
        lines_with_step = []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                clean = strip_ansi(raw_line)
                if "step:" in clean:
                    lines_with_step.append(clean.strip()[:250])
        print(f"[DEBUG] Lines containing 'step:': {len(lines_with_step)}")
        for i, l in enumerate(lines_with_step[:5]):
            print(f"  [{i}] {l}")
        if not lines_with_step:
            print("[DEBUG] First 10 non-empty lines:")
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                shown = 0
                for raw_line in f:
                    line = raw_line.strip()
                    if line:
                        print(f"  {strip_ansi(line)[:200]}")
                        shown += 1
                        if shown >= 10:
                            break
    return records


def build_series(records):
    """从 records 中提取 steps 和 series，缺失值填 NaN 保证长度一致。"""
    N = len(records)
    steps = np.array([r["step"] for r in records], dtype=int)
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    all_keys.discard("step")

    series = defaultdict(list)
    for key in all_keys:
        for r in records:
            series[key].append(float(r[key]) if key in r else np.nan)
    return steps, series, all_keys


def plot_panels(records, out_dir: str):
    """按 PANELS 分组画图，每个面板一张图。"""
    if not records:
        print("[WARN] No records found.")
        return

    steps, series, all_keys = build_series(records)

    os.makedirs(out_dir, exist_ok=True)

    for panel_name, key_list in PANELS.items():
        existing = [k for k in key_list if k in all_keys]
        if not existing:
            continue

        n_plots = len(existing)
        n_cols = min(3, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False
        )
        fig.suptitle(panel_name, fontsize=14, fontweight="bold")

        for idx, metric in enumerate(existing):
            ax = axes[idx // n_cols][idx % n_cols]
            values = np.array(series[metric])
            ax.plot(steps, values, linewidth=0.8, color="#1f77b4")
            ax.set_title(metric, fontsize=9)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            # 平滑趋势线
            if len(values) > 20:
                window = max(5, len(values) // 30)
                if window > 1:
                    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
                    ax.plot(steps[window - 1:], smoothed, linewidth=1.5, color="#d62728",
                            alpha=0.7, label=f"MA-{window}")
                    ax.legend(fontsize=7)

        # 隐藏多余的子图
        for idx in range(n_plots, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        fig.tight_layout()
        safe_name = panel_name.replace(" ", "_").replace("&", "and").replace("/", "_")
        path = os.path.join(out_dir, f"{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {path}")


def plot_combined(records, out_dir: str):
    """画一张综合仪表盘，6个最关键的指标。"""
    if not records:
        return

    steps, series, all_keys = build_series(records)

    combined = [
        ("critic/score/mean",       "Score Mean",        "tab:green"),
        ("env/finish_ratio",         "Finish Ratio",      "tab:blue"),
        ("env/ratio_of_valid_action","Valid Action Ratio", "tab:cyan"),
        ("actor/pg_loss",            "PG Loss",           "tab:red"),
        ("actor/grad_norm",          "Grad Norm",         "tab:orange"),
        ("actor/kl_loss",            "KL Loss",           "tab:purple"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Training Dashboard", fontsize=16, fontweight="bold")

    for idx, (metric, label, color) in enumerate(combined):
        ax = axes[idx // 3][idx % 3]
        if metric not in series:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue
        values = np.array(series[metric])
        ax.plot(steps, values, linewidth=0.6, color=color, alpha=0.6)
        # 平滑线
        if len(values) > 10:
            w = max(3, len(values) // 25)
            if w > 1:
                smoothed = np.convolve(values, np.ones(w) / w, mode="valid")
                ax.plot(steps[w - 1:], smoothed, linewidth=2, color=color)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "00_Dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from Search-R1 log.")
    parser.add_argument("logfile", type=str, help="Path to the training log file")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: <logfile_dir>/plots)")
    parser.add_argument("--step", type=int, nargs=2, metavar=("START", "END"),
                        help="Only plot steps in [START, END]")
    parser.add_argument("--live", action="store_true",
                        help="Live refresh mode (every 30s)")

    args = parser.parse_args()

    if args.out is None:
        logdir = os.path.dirname(os.path.abspath(args.logfile)) or "."
        args.out = os.path.join(logdir, "plots")

    step_range = tuple(args.step) if args.step else None

    if args.live:
        print(f"[LIVE] Refreshing every 30s, output to {args.out}")
        while True:
            records = parse_log(args.logfile, step_range)
            plot_panels(records, args.out)
            plot_combined(records, args.out)
            print(f"[LIVE] Updated at step {records[-1]['step'] if records else 'N/A'}")
            time.sleep(30)
    else:
        records = parse_log(args.logfile, step_range)
        print(f"Parsed {len(records)} steps from {args.logfile}")
        plot_panels(records, args.out)
        plot_combined(records, args.out)
        print(f"\nAll plots saved to: {args.out}")


if __name__ == "__main__":
    main()
