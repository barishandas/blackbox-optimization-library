"""
BenchmarkRunner: orchestrates running one or more algorithms against one or more
benchmark functions over multiple repeated trials, optionally measuring energy.
"""
from __future__ import annotations
import csv
import time
import yaml
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm

from .algorithms.base import BBOAlgorithm, RunResult
from .benchmarks.functions import BenchmarkFunction
from .energy.monitor import EnergyMonitor, EnergyReport
from .metrics import summary_table


@dataclass
class TrialRecord:
    algorithm: str
    function: str
    dimension: int
    budget: int
    seed: int
    best_f: float
    n_evaluations: int
    wall_time_s: float
    energy_joules: float | None
    energy_method: str


class BenchmarkRunner:
    """
    Run a cross-product of algorithms × functions × dimensions × seeds.

    Parameters
    ----------
    measure_energy : wrap each run in EnergyMonitor (adds ~0.25 s overhead per run)
    results_dir    : if given, save raw_data.csv and metadata.yaml here after run
    """

    def __init__(
        self,
        measure_energy: bool = False,
        results_dir: str | Path | None = None,
    ):
        self.measure_energy = measure_energy
        self.results_dir = Path(results_dir) if results_dir else None
        self._records: list[TrialRecord] = []
        self._histories: dict[tuple, list[RunResult]] = {}

    def run(
        self,
        algorithms: Sequence[BBOAlgorithm],
        functions: Sequence[BenchmarkFunction],
        dimensions: Sequence[int],
        budget: int,
        n_runs: int = 10,
        seeds: Sequence[int] | None = None,
    ) -> None:
        if seeds is None:
            seeds = list(range(n_runs))

        total = len(algorithms) * len(functions) * len(dimensions) * len(seeds)
        bar = tqdm(total=total, desc="Benchmarking", unit="run")

        for algo in algorithms:
            for fn in functions:
                for d in dimensions:
                    bounds = fn.bounds(d)
                    key = (algo.name, fn.name, d)
                    self._histories.setdefault(key, [])

                    for seed in seeds:
                        t0 = time.perf_counter()

                        if self.measure_energy:
                            with EnergyMonitor() as em:
                                result = algo.minimize(fn, bounds, budget=budget, seed=seed)
                            energy_j = em.report.best_estimate_joules
                            energy_method = em.report.measurement_method
                        else:
                            result = algo.minimize(fn, bounds, budget=budget, seed=seed)
                            energy_j = None
                            energy_method = "not measured"

                        wall = time.perf_counter() - t0
                        self._histories[key].append(result)
                        self._records.append(TrialRecord(
                            algorithm=algo.name,
                            function=fn.name,
                            dimension=d,
                            budget=budget,
                            seed=seed,
                            best_f=result.best_f,
                            n_evaluations=result.n_evaluations,
                            wall_time_s=round(wall, 4),
                            energy_joules=round(energy_j, 6) if energy_j else None,
                            energy_method=energy_method,
                        ))
                        bar.update()

        bar.close()

        if self.results_dir:
            self._save(budget)

    # ------------------------------------------------------------------
    # Results access
    # ------------------------------------------------------------------

    def get_results(
        self,
        algorithm: str,
        function: str,
        dimension: int,
    ) -> list[RunResult]:
        return self._histories.get((algorithm, function, dimension), [])

    def summary(
        self,
        function: str,
        dimension: int,
        target: float,
        budget: int,
    ) -> dict[str, dict[str, float]]:
        algo_names = {r.algorithm for r in self._records if r.function == function and r.dimension == dimension}
        results_by_algo = {
            name: self._histories[(name, function, dimension)]
            for name in algo_names
            if (name, function, dimension) in self._histories
        }
        return summary_table(results_by_algo, target, budget)

    def print_summary(
        self,
        function: str,
        dimension: int,
        target: float,
        budget: int,
    ) -> None:
        table = self.summary(function, dimension, target, budget)
        header = f"{'Algorithm':<20} {'Success%':>9} {'ERT':>12} {'Median FEs':>12} {'AUC':>10}"
        print(f"\n=== {function} | d={dimension} | target={target:.2e} ===")
        print(header)
        print("-" * len(header))
        for name, m in table.items():
            sr = f"{m['success_rate']*100:.1f}%"
            ert = f"{m['ert']:.0f}" if m['ert'] != float("inf") else "inf"
            med = f"{m['median_feval']:.0f}" if m['median_feval'] != float("inf") else "inf"
            auc = f"{m['auc']:.4f}"
            print(f"{name:<20} {sr:>9} {ert:>12} {med:>12} {auc:>10}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, budget: int) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # raw_data.csv
        csv_path = self.results_dir / "raw_data.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self._records[0])))
            writer.writeheader()
            for rec in self._records:
                writer.writerow(asdict(rec))

        # metadata.yaml
        meta = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor(),
            "budget": budget,
            "n_trials": len(self._records),
            "algorithms": list({r.algorithm for r in self._records}),
            "functions": list({r.function for r in self._records}),
            "dimensions": list({r.dimension for r in self._records}),
        }
        with open(self.results_dir / "metadata.yaml", "w") as f:
            yaml.dump(meta, f, default_flow_style=False)

        print(f"Results saved to {self.results_dir}/")
