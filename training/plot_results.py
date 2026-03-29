#!/usr/bin/env python3
"""
Plot training curves from phase3_scores.json.

Produces:
  - training/training_curves.png  (raw scores + rolling avg + epsilon decay)

Usage:
    python3 adaptive_cyber_defense/training/plot_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("[error] matplotlib not installed — run: pip install matplotlib")
    sys.exit(1)

_HERE      = Path(__file__).resolve().parent
SCORES_IN  = _HERE / "phase3_scores.json"
PLOT_OUT   = _HERE / "training_curves.png"


def rolling(scores: list[float], w: int) -> list[float]:
    return [
        sum(scores[max(0, i - w + 1): i + 1]) / min(i + 1, w)
        for i in range(len(scores))
    ]


def epsilon_curve(n: int, start: float = 1.0, decay: float = 0.995, floor: float = 0.05) -> list[float]:
    eps, curve = start, []
    for _ in range(n):
        curve.append(eps)
        eps = max(floor, eps * decay)
    return curve


def main() -> None:
    if not SCORES_IN.exists():
        print(f"[error] {SCORES_IN} not found — run train_phase3.py first")
        sys.exit(1)

    data   = json.loads(SCORES_IN.read_text())
    scores = data["all_scores"]
    n      = len(scores)
    episodes = list(range(1, n + 1))
    roll50   = rolling(scores, 50)
    roll10   = rolling(scores, 10)
    epsilons = epsilon_curve(n)

    # ── eval bar chart data ───────────────────────────────────────────
    eval_d   = data.get("eval", {})
    agents   = ["QL Agent", "Baseline", "Ignore"]
    avgs     = [
        eval_d.get("ql_avg",   0),
        eval_d.get("base_avg", 0),
        eval_d.get("ign_avg",  0),
    ]
    colors   = ["#2ea8e0", "#27ae60", "#e74c3c"]

    # ── layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9), facecolor="#0d1117")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, :])   # full-width top  — training curve
    ax2 = fig.add_subplot(gs[1, 0])   # bottom-left      — epsilon decay
    ax3 = fig.add_subplot(gs[1, 1])   # bottom-right     — eval bar chart

    _bg  = "#0d1117"
    _fg  = "#c9d1d9"
    _grid = "#21262d"

    def _style(ax, title: str) -> None:
        ax.set_facecolor(_bg)
        ax.tick_params(colors=_fg, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(_grid)
        ax.set_title(title, color=_fg, fontsize=10, pad=6)
        ax.xaxis.label.set_color(_fg)
        ax.yaxis.label.set_color(_fg)
        ax.grid(color=_grid, linewidth=0.5)

    # ── ax1: raw + rolling averages ───────────────────────────────────
    ax1.plot(episodes, scores,  color="#30363d", linewidth=0.6, label="Raw score",    zorder=1)
    ax1.plot(episodes, roll10,  color="#58a6ff", linewidth=1.0, label="Roll-avg 10",  zorder=2)
    ax1.plot(episodes, roll50,  color="#f0c030", linewidth=2.0, label="Roll-avg 50",  zorder=3)

    # phase boundary lines
    for boundary, label in [(50, "P1→P2"), (200, "P2→P3")]:
        ax1.axvline(boundary, color="#8b949e", linestyle="--", linewidth=0.8)
        ax1.text(boundary + 3, max(scores) * 0.95, label, color="#8b949e", fontsize=7)

    ax1.set_xlim(1, n)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score [0–1]")
    ax1.legend(facecolor=_bg, edgecolor=_grid, labelcolor=_fg, fontsize=8, loc="upper left")
    _style(ax1, f"Training Curve — {n} episodes (Hard difficulty)")

    # phase avg annotations
    p_avgs = data.get("phase_avgs", {})
    for (x0, x1), key, label in [((0, 50), "p1", "P1"), ((50, 200), "p2", "P2"), ((200, n), "p3", "P3")]:
        avg = p_avgs.get(key, 0)
        ax1.hlines(avg, x0 + 1, x1, colors="#39d353", linewidths=1.2, linestyles="dotted")
        ax1.text(x1 - 10, avg + 0.02, f"{label} avg\n{avg:.3f}", color="#39d353", fontsize=7, ha="right")

    # ── ax2: epsilon decay ────────────────────────────────────────────
    ax2.plot(episodes, epsilons, color="#da8b45", linewidth=1.5)
    ax2.axhline(0.05, color="#8b949e", linestyle=":", linewidth=0.8)
    ax2.text(n * 0.6, 0.07, "floor ε=0.05", color="#8b949e", fontsize=7)
    ax2.set_xlim(1, n)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("ε (exploration rate)")
    _style(ax2, "Epsilon Decay")

    # ── ax3: eval bar chart ───────────────────────────────────────────
    bars = ax3.bar(agents, avgs, color=colors, width=0.5, zorder=2)
    for bar, avg in zip(bars, avgs):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            avg + 0.01,
            f"{avg:.4f}",
            ha="center", va="bottom",
            color=_fg, fontsize=8,
        )
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("Avg Score (20 seeds)")
    _style(ax3, "Eval: QL vs Baseline vs Ignore")

    fig.suptitle(
        "Adaptive Cyber Defense — Q-Learning Training Summary",
        color=_fg, fontsize=12, y=0.98,
    )

    fig.savefig(PLOT_OUT, dpi=150, bbox_inches="tight", facecolor=_bg)
    print(f"Saved plot → {PLOT_OUT}")


if __name__ == "__main__":
    main()
