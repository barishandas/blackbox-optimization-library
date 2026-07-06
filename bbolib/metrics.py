"""
Performance metrics for BBO benchmark runs.

All functions operate on lists of RunResult objects from the same
(algorithm, function, dimension, budget) configuration.
"""
import numpy as np
from .algorithms.base import RunResult


def success_rate(results: list[RunResult], target: float) -> float:
    """Fraction of runs that reached `target` quality."""
    successes = sum(1 for r in results if r.best_f <= target)
    return successes / len(results)


def expected_running_time(results: list[RunResult], target: float) -> float:
    """
    ERT = (total evaluations across all runs) / (number of successful runs).
    Returns inf if no run succeeded.
    """
    successful = [r for r in results if r.best_f <= target]
    if not successful:
        return float("inf")
    total_evals = sum(r.n_evaluations for r in results)
    return total_evals / len(successful)


def median_feval_to_target(results: list[RunResult], target: float) -> float:
    """
    Median number of evaluations to first reach `target` across successful runs.
    Returns inf if fewer than half the runs succeeded.
    """
    hitting_times = []
    for r in results:
        for i, f in enumerate(r.history_f):
            if f <= target:
                hitting_times.append(i + 1)
                break
    if not hitting_times:
        return float("inf")
    return float(np.median(hitting_times))


def auc_convergence(results: list[RunResult], budget: int, log_scale: bool = True) -> float:
    """
    Area under the mean convergence curve (lower is better).
    Pads history to `budget` with the final value.
    If log_scale=True, applies log10 to function values before integration.
    """
    curves = []
    for r in results:
        hist = list(r.history_f)
        if len(hist) < budget:
            hist += [hist[-1]] * (budget - len(hist))
        hist = hist[:budget]
        curves.append(hist)

    mean_curve = np.mean(curves, axis=0)
    if log_scale:
        mean_curve = np.log10(np.clip(mean_curve, 1e-12, None))

    return float(np.trapz(mean_curve) / budget)


def convergence_curve(results: list[RunResult], budget: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (x, median_curve, [q25, q75]) padded to `budget` evaluations.
    Useful for plotting.
    """
    curves = []
    for r in results:
        hist = list(r.history_f)
        if len(hist) < budget:
            hist += [hist[-1]] * (budget - len(hist))
        curves.append(hist[:budget])

    arr = np.array(curves)
    x = np.arange(1, budget + 1)
    return x, np.median(arr, axis=0), np.percentile(arr, [25, 75], axis=0)


def summary_table(
    results_by_algo: dict[str, list[RunResult]],
    target: float,
    budget: int,
) -> dict[str, dict[str, float]]:
    """
    Build a summary dict keyed by algorithm name with metrics:
      success_rate, ert, median_feval, auc
    """
    table = {}
    for algo_name, results in results_by_algo.items():
        table[algo_name] = {
            "success_rate": success_rate(results, target),
            "ert": expected_running_time(results, target),
            "median_feval": median_feval_to_target(results, target),
            "auc": auc_convergence(results, budget),
        }
    return table
