#!/usr/bin/env python3
"""Generate presentation-ready figures for the 27 Sep--02 Oct RA-QMIX update.

The script reads the archived experiment summaries where available and exports
each figure as a high-resolution PNG and an editable SVG.  All figures use a
white background so they can be dropped directly into the Canva report.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ra_qmix_mplconfig")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ra_qmix_cache")
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

WHITE = "#FFFFFF"
INK = "#20242B"
MUTED = "#667085"
GRID = "#E6E9EF"
BLUE = "#4C6FFF"
PURPLE = "#8B5CF6"
TEAL = "#159A9C"
ORANGE = "#E58A3A"
RED = "#D95D5D"
PALE_GREEN = "#EAF7F1"


def load_summary(run: str) -> dict:
    path = ROOT / "lab_results" / run / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.color": INK,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def finish(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def add_title(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", color=INK, pad=30)
    ax.text(
        0,
        1.035,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11.5,
        color=MUTED,
    )


def label_bars(ax: plt.Axes, bars, values: list[float], decimals: int = 1) -> None:
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for bar, value in zip(bars, values, strict=True):
        offset = 0.018 * span
        y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:.{decimals}f}",
            ha="center",
            va=va,
            fontsize=9.5,
            fontweight="bold",
            color=INK,
        )


def make_ablation_chart() -> None:
    summary = load_summary("ra_qmix_leave_one_out_5seed_full_cuda_20260927T095633Z")
    scenario_keys = [
        "in_distribution",
        "early_failure",
        "slow_service",
        "combined_pressure",
    ]
    scenario_labels = ["Nominal", "Early failure", "Slow service", "Combined pressure"]
    series = [
        ("edge_ablation_minus_full", "Remove edge", BLUE),
        ("queue_ablation_minus_full", "Remove queue mixer", PURPLE),
        ("cf_ablation_minus_full", "Remove CF loss", TEAL),
        ("qmix_minus_full", "Standard QMIX", ORANGE),
    ]

    rows = {
        (row["contrast"], row["scenario"]): float(row["delta_mean"])
        for row in summary["paired_differences"]
        if row["row_type"] == "across_training_seeds" and row["budget"] == 50000
    }

    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    x = np.arange(len(scenario_keys))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    for offset, (key, label, color) in zip(offsets, series, strict=True):
        values = [rows[(key, scenario)] for scenario in scenario_keys]
        bars = ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=color,
            label=label,
            zorder=3,
        )
        label_bars(ax, bars, values)

    ax.axhline(0, color=INK, linewidth=1.2, zorder=4)
    ax.set_ylim(-18.5, 19.5)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
    ax.set_xticks(x, scenario_labels)
    ax.set_ylabel("Cost delta vs Full RA-QMIX")
    ax.text(
        1.0,
        -0.16,
        "Positive = Full RA-QMIX has lower cost",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color=MUTED,
    )
    add_title(
        ax,
        "Ablation effects change across operating conditions",
        "Mean paired cost difference across 5 training seeds at 50k steps",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=10.5,
        handlelength=1.1,
        columnspacing=1.5,
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, pad=8)
    finish(fig, "01_ablation_deltas")


def make_tradeoff_chart() -> None:
    # Values are the cross-seed results summarized in the research update.
    points = [
        ("Domain randomization", 33.54, 0.008, RED),
        ("Fixed 87.5% nominal", 10.44, 0.542, ORANGE),
        ("Curriculum + replay", -1.66, 0.607, TEAL),
        ("Curriculum + KL", -5.68, 0.550, PURPLE),
    ]

    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    ax.axvspan(-10, 5, color=PALE_GREEN, zorder=0)
    ax.axvline(5, color=TEAL, linewidth=1.5, linestyle=(0, (5, 4)), zorder=2)
    ax.text(4.4, 0.685, "Nominal gate ≤ 5%", ha="right", va="top", color=TEAL, fontsize=10.5)

    label_offsets = {
        "Domain randomization": (-8, 14, "right"),
        "Fixed 87.5% nominal": (12, -24, "left"),
        "Curriculum + replay": (12, 12, "left"),
        "Curriculum + KL": (12, -23, "left"),
    }
    for label, nominal, stress, color in points:
        ax.scatter(
            nominal,
            stress,
            s=190,
            color=color,
            edgecolor=WHITE,
            linewidth=2.2,
            zorder=4,
        )
        dx, dy, ha = label_offsets[label]
        ax.annotate(
            label,
            (nominal, stress),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=11,
            fontweight="bold",
            color=INK,
        )

    ax.set_xlim(-10, 38)
    ax.set_ylim(-0.045, 0.71)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(color=GRID, linewidth=0.9, zorder=0)
    ax.set_xlabel("Nominal cost degradation (lower is better)", labelpad=12)
    ax.set_ylabel("Stress max-regret improvement (higher is better)", labelpad=12)
    add_title(
        ax,
        "Curriculum training resolves the stress–nominal trade-off",
        "Curriculum arms enter the nominal-preservation region while retaining the stress gain",
    )
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=8)
    finish(fig, "02_training_tradeoff")


def make_curriculum_chart() -> None:
    summary = load_summary("ra_qmix_anchor_preservation_full_cpu_20260930T075415Z")
    results = {
        row["candidate_regime"]: row
        for row in summary["primary_results"]
        if row["row_type"] == "across_training_seeds"
    }
    keys = ["fixed_87_5", "curriculum_replay", "curriculum_kl_anchor"]
    labels = ["Fixed mix", "Curriculum + replay", "Curriculum + KL"]
    stress = [int(results[key]["stress_better_count"]) for key in keys]
    nominal = [int(results[key]["nominal_within_10pct_count"]) for key in keys]

    fig, ax = plt.subplots(figsize=(12.8, 7.2), constrained_layout=True)
    x = np.arange(len(keys))
    width = 0.30
    stress_bars = ax.bar(x - width / 2, stress, width, color=BLUE, label="Stress better than control", zorder=3)
    nominal_bars = ax.bar(x + width / 2, nominal, width, color=PURPLE, label="Nominal within 10%", zorder=3)

    ax.axhline(4, color=TEAL, linewidth=1.5, linestyle=(0, (5, 4)), zorder=2)
    ax.text(2.43, 4.06, "Gate: 4/5", ha="right", va="bottom", color=TEAL, fontsize=10.5)
    ax.set_ylim(0, 5.7)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Training seeds passing the gate")

    for bars, values in ((stress_bars, stress), (nominal_bars, nominal)):
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.10,
                f"{value}/5",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
                color=INK,
            )

    add_title(
        ax,
        "Both curriculum variants pass the development gates",
        "Five-seed development panel; curriculum + replay was selected for fresh-seed confirmation",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=11)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, pad=8)
    finish(fig, "03_curriculum_seed_gates")


def main() -> None:
    apply_style()
    make_ablation_chart()
    make_tradeoff_chart()
    make_curriculum_chart()
    print(f"Wrote 3 PNG and 3 SVG charts to {OUT}")


if __name__ == "__main__":
    main()
