"""
Full benchmark run: all algorithms × all functions × multiple dimensions.
Saves results to results/full_run/ and generates comparison plots.

Usage:
    python scripts/run_full_benchmark.py [--budget 10000] [--runs 10] [--energy]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt

from bbolib import BenchmarkRunner, CMAESAlgorithm, DEAlgorithm, PSOAlgorithm, RandomSearch
from bbolib.benchmarks import list_functions, get_function
from bbolib.plotting import plot_convergence, plot_summary_bar

parser = argparse.ArgumentParser()
parser.add_argument("--budget", type=int, default=10_000)
parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--dims", nargs="+", type=int, default=[5, 10, 20])
parser.add_argument("--energy", action="store_true")
parser.add_argument("--fns", nargs="+", default=None,
                    help="Subset of functions (default: all)")
parser.add_argument("--out", default="results/full_run")
args = parser.parse_args()

algos = [CMAESAlgorithm(), DEAlgorithm(), PSOAlgorithm(), RandomSearch()]
fn_names = args.fns or list_functions()
fns = [get_function(n) for n in fn_names]

print(f"Running {len(algos)} algorithms × {len(fns)} functions × {len(args.dims)} dims × {args.runs} runs")
print(f"Budget per run: {args.budget} evaluations")

runner = BenchmarkRunner(measure_energy=args.energy, results_dir=args.out)
runner.run(algos, fns, dimensions=args.dims, budget=args.budget, n_runs=args.runs)

# Print summary for each config
TARGET = 1e-4
for fn in fns:
    for d in args.dims:
        runner.print_summary(fn.name, d, target=TARGET, budget=args.budget)

# Save comparison plots
plot_dir = Path(args.out) / "plots"
plot_dir.mkdir(parents=True, exist_ok=True)

for fn in fns:
    for d in args.dims:
        results_by_algo = {
            a.name: runner.get_results(a.name, fn.name, d)
            for a in algos
            if runner.get_results(a.name, fn.name, d)
        }
        if not results_by_algo:
            continue

        fig = plot_convergence(
            results_by_algo,
            budget=args.budget,
            title=f"{fn.name} | d={d}",
            out_path=plot_dir / f"convergence_{fn.name}_d{d}.png",
        )
        plt.close(fig)

        fig = plot_summary_bar(
            results_by_algo,
            target=TARGET,
            budget=args.budget,
            metric="success_rate",
            title=f"Success Rate — {fn.name} | d={d}",
            out_path=plot_dir / f"success_{fn.name}_d{d}.png",
        )
        plt.close(fig)

print(f"\nPlots saved to {plot_dir}/")
