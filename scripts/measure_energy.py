"""
Measure and compare energy consumption of algorithms on a single function.
Outputs a scatter plot of energy (J) vs solution quality.

Usage:
    python scripts/measure_energy.py --fn rastrigin --dim 10 --budget 5000 --runs 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--fn", default="rastrigin")
parser.add_argument("--dim", type=int, default=10)
parser.add_argument("--budget", type=int, default=5_000)
parser.add_argument("--runs", type=int, default=5)
parser.add_argument("--out", default="energy_vs_quality.png")
args = parser.parse_args()

import matplotlib.pyplot as plt
from bbolib import BenchmarkRunner, CMAESAlgorithm, DEAlgorithm, PSOAlgorithm, RandomSearch, get_function
from bbolib.plotting import plot_energy_vs_quality

algos = [CMAESAlgorithm(), DEAlgorithm(), PSOAlgorithm(), RandomSearch()]
fn = get_function(args.fn)

runner = BenchmarkRunner(measure_energy=True)
runner.run(algos, [fn], dimensions=[args.dim], budget=args.budget, n_runs=args.runs)

# Pull energy and quality per algorithm
import csv
import pandas as pd

# Reconstruct from records
records = runner._records
df_rows = [
    {
        "algorithm": r.algorithm,
        "best_f": r.best_f,
        "energy_joules": r.energy_joules or 0.0,
    }
    for r in records
]

import pandas as pd
df = pd.DataFrame(df_rows)
print("\n=== Energy summary ===")
print(df.groupby("algorithm")[["best_f", "energy_joules"]].describe())

energy_by_algo = df.groupby("algorithm")["energy_joules"].apply(list).to_dict()
quality_by_algo = df.groupby("algorithm")["best_f"].apply(list).to_dict()

fig = plot_energy_vs_quality(
    energy_by_algo,
    quality_by_algo,
    title=f"Energy vs Quality — {args.fn} | d={args.dim}",
    out_path=args.out,
)
print(f"\nPlot saved to {args.out}")
plt.show()
