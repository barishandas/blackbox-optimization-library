"""
Plotting utilities for BBO benchmark results.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from .algorithms.base import RunResult
from .metrics import convergence_curve, summary_table


def plot_convergence(
    results_by_algo: dict[str, list[RunResult]],
    budget: int,
    title: str = "Convergence",
    log_y: bool = True,
    out_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot median convergence curves with 25-75th percentile bands.

    Parameters
    ----------
    results_by_algo : dict mapping algorithm name -> list of RunResult
    budget          : x-axis limit (number of function evaluations)
    log_y           : use log scale on y-axis
    out_path        : if given, save figure here
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (name, results) in enumerate(results_by_algo.items()):
        color = colors[idx % len(colors)]
        x, med, pct = convergence_curve(results, budget)
        ax.plot(x, med, label=name, color=color, linewidth=1.8)
        ax.fill_between(x, pct[0], pct[1], alpha=0.15, color=color)

    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best f (lower = better)")
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.legend(framealpha=0.7)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def plot_summary_bar(
    results_by_algo: dict[str, list[RunResult]],
    target: float,
    budget: int,
    metric: str = "success_rate",
    title: str | None = None,
    out_path: str | Path | None = None,
) -> plt.Figure:
    """
    Bar chart comparing a single metric across algorithms.

    metric: one of 'success_rate', 'ert', 'median_feval', 'auc'
    """
    table = summary_table(results_by_algo, target, budget)
    names = list(table)
    values = [table[n][metric] for n in names]

    fig, ax = plt.subplots(figsize=(max(5, len(names) * 1.2), 4))
    bars = ax.bar(names, values)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or metric.replace("_", " ").title())
    ax.bar_label(bars, fmt="%.3g", padding=3)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def plot_energy_vs_quality(
    energy_by_algo: dict[str, list[float]],   # joules per run
    quality_by_algo: dict[str, list[float]],  # best_f per run
    title: str = "Energy vs Solution Quality",
    out_path: str | Path | None = None,
) -> plt.Figure:
    """
    Scatter plot of energy consumed vs solution quality per run.
    Pareto-efficient algorithms appear in the lower-left corner.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, name in enumerate(energy_by_algo):
        color = colors[idx % len(colors)]
        x = energy_by_algo[name]
        y = quality_by_algo[name]
        ax.scatter(x, y, label=name, color=color, alpha=0.7, s=40)
        ax.scatter(np.mean(x), np.mean(y), marker="*", color=color, s=200, zorder=5)

    ax.set_xlabel("Energy (J)")
    ax.set_ylabel("Best f value (lower = better)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig
