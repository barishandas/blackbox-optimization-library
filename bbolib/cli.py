"""
Command-line interface.

  bbo-run   --algos cma-es de pso --fns sphere rastrigin --dim 10 --budget 5000 --runs 5
  bbo-compare --results-dir results/my_run --target 1e-6
"""
import argparse
import sys
from pathlib import Path

from .algorithms import CMAESAlgorithm, DEAlgorithm, PSOAlgorithm, RandomSearch, SPSAAlgorithm, FDAlgorithm, PEPGAlgorithm
from .benchmarks import get_function
from .runner import BenchmarkRunner
from .plotting import plot_convergence, plot_summary_bar

_ALGO_MAP = {
    "cma-es": CMAESAlgorithm,
    "de": DEAlgorithm,
    "pso": PSOAlgorithm,
    "random": RandomSearch,
    "spsa": SPSAAlgorithm,
    "fd": FDAlgorithm,
    "pepg": PEPGAlgorithm,
}


def run():
    parser = argparse.ArgumentParser(prog="bbo-run", description="Run BBO benchmarks")
    parser.add_argument("--algos", nargs="+", default=["cma-es", "de", "pso"],
                        help="Algorithms to benchmark (choices: cma-es de pso random)")
    parser.add_argument("--fns", nargs="+", default=["sphere", "rastrigin", "rosenbrock"],
                        help="Benchmark functions")
    parser.add_argument("--dim", nargs="+", type=int, default=[10],
                        help="Problem dimensions")
    parser.add_argument("--budget", type=int, default=10_000,
                        help="Function evaluation budget per run")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of independent runs per configuration")
    parser.add_argument("--energy", action="store_true",
                        help="Measure energy consumption (Linux RAPL or psutil estimate)")
    parser.add_argument("--out", default=None,
                        help="Directory to save results (CSV + metadata)")
    parser.add_argument("--plot", action="store_true",
                        help="Show convergence plots after benchmarking")
    parser.add_argument("--target", type=float, default=1e-6,
                        help="Target function value for success/ERT metrics")
    args = parser.parse_args()

    algos = []
    for name in args.algos:
        if name not in _ALGO_MAP:
            print(f"Unknown algorithm '{name}'. Options: {list(_ALGO_MAP)}", file=sys.stderr)
            sys.exit(1)
        algos.append(_ALGO_MAP[name]())

    fns = []
    for name in args.fns:
        try:
            fns.append(get_function(name))
        except KeyError as e:
            print(e, file=sys.stderr)
            sys.exit(1)

    runner = BenchmarkRunner(
        measure_energy=args.energy,
        results_dir=args.out,
    )
    runner.run(algos, fns, args.dim, budget=args.budget, n_runs=args.runs)

    for fn in fns:
        for d in args.dim:
            runner.print_summary(fn.name, d, target=args.target, budget=args.budget)

            if args.plot:
                algo_names = [a.name for a in algos]
                results_by_algo = {
                    name: runner.get_results(name, fn.name, d)
                    for name in algo_names
                }
                fig = plot_convergence(
                    results_by_algo,
                    budget=args.budget,
                    title=f"{fn.name} | d={d}",
                )
                fig.show()

    if args.plot:
        import matplotlib.pyplot as plt
        plt.show()


def compare():
    parser = argparse.ArgumentParser(prog="bbo-compare", description="Compare saved results")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--target", type=float, default=1e-6)
    parser.add_argument("--metric", default="success_rate",
                        choices=["success_rate", "ert", "median_feval", "auc"])
    parser.add_argument("--out", default=None, help="Save plot to this path")
    args = parser.parse_args()

    import csv
    import pandas as pd

    csv_path = Path(args.results_dir) / "raw_data.csv"
    if not csv_path.exists():
        print(f"No raw_data.csv found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(df.groupby(["algorithm", "function", "dimension"])["best_f"].describe().to_string())
