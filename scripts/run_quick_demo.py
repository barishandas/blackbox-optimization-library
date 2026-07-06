"""
Quick demo: run CMA-ES, DE, PSO, and Random Search on Sphere and Rastrigin
in d=10 with 5000 evaluations, print a summary, and show convergence plots.

Usage:
    python scripts/run_quick_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt

from bbolib import BenchmarkRunner, CMAESAlgorithm, DEAlgorithm, PSOAlgorithm, RandomSearch, get_function
from bbolib.plotting import plot_convergence, plot_summary_bar

BUDGET = 50_000
DIM = 10
N_RUNS = 5
TARGET = 1e-6

algos = [CMAESAlgorithm(), DEAlgorithm(), PSOAlgorithm(), RandomSearch()]
fns = [get_function("sphere"), get_function("rastrigin"), get_function("rosenbrock")]

runner = BenchmarkRunner(measure_energy=False)
runner.run(algos, fns, dimensions=[DIM], budget=BUDGET, n_runs=N_RUNS)

# Print summary tables
for fn in fns:
    runner.print_summary(fn.name, DIM, target=TARGET, budget=BUDGET)

# Convergence plots
fig, axes = plt.subplots(1, len(fns), figsize=(5 * len(fns), 4))
for ax, fn in zip(axes, fns):
    results_by_algo = {
        a.name: runner.get_results(a.name, fn.name, DIM)
        for a in algos
    }
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (name, results) in enumerate(results_by_algo.items()):
        from bbolib.metrics import convergence_curve
        x, med, pct = convergence_curve(results, BUDGET)
        ax.semilogy(x, med, label=name, color=colors[idx], linewidth=1.8)
        ax.fill_between(x, pct[0], pct[1], alpha=0.12, color=colors[idx])
    ax.set_title(fn.name)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best f")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

plt.suptitle(f"Convergence comparison | d={DIM} | budget={BUDGET}")
plt.tight_layout()
plt.savefig("convergence_demo.png", dpi=150)
print("\nConvergence plot saved to convergence_demo.png")
plt.show()
