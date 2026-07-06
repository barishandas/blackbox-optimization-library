# Black-Box Optimization Algorithm Library

A continuously updated library for benchmarking **energy efficiency**, **computational complexity**, and **convergence performance** of black-box optimization (BBO) algorithms.

## What This Is

Black-box optimizers treat the objective function as a black box — no gradients, no structure assumed. This library provides:

- **Runnable algorithm wrappers** (CMA-ES, DE, PSO, Random Search)
- **14 standard benchmark functions** (Sphere → Michalewicz, grouped by difficulty)
- **Benchmark runner** with multi-algorithm × multi-function × multi-dimension cross-product
- **Energy monitor** (Linux RAPL; psutil CPU-time estimate on other platforms)
- **Metrics**: success rate, ERT, median FEs to target, AUC convergence
- **Plots**: convergence curves with percentile bands, success rate bars, energy-vs-quality scatter
- **CLI**: `bbo-run` and `bbo-compare` commands
- **Algorithm docs** with complexity tables, convergence data, and references

## Quick Start

```bash
pip install -e ".[dev]"

# Quick demo (CMA-ES, DE, PSO, Random on Sphere + Rastrigin, d=10)
python scripts/run_quick_demo.py

# CLI: run and print summary table
bbo-run --algos cma-es de pso --fns sphere rastrigin --dim 10 --budget 5000 --runs 5

# Full benchmark across all functions and dimensions
python scripts/run_full_benchmark.py --budget 10000 --runs 10 --dims 5 10 20 --out results/run1

# Energy measurement run
python scripts/measure_energy.py --fn rastrigin --dim 10 --budget 5000 --runs 5

# Run tests
pytest tests/ -v
```

## Repository Structure

```
blackbox-optimization-library/
├── bbolib/                  # Python package
│   ├── algorithms/          # CMA-ES, DE, PSO, Random Search wrappers
│   │   ├── base.py          # BBOAlgorithm ABC + RunResult dataclass
│   │   ├── cma_es.py        # Wraps `cma` library
│   │   ├── de.py            # Wraps scipy differential_evolution
│   │   ├── pso.py           # Pure-Python PSO implementation
│   │   └── random_search.py # Uniform random baseline
│   ├── benchmarks/
│   │   └── functions.py     # 14 standard test functions
│   ├── energy/
│   │   └── monitor.py       # EnergyMonitor context manager (RAPL + psutil)
│   ├── runner.py            # BenchmarkRunner: multi-algo × multi-fn orchestration
│   ├── metrics.py           # success_rate, ERT, AUC, convergence_curve
│   ├── plotting.py          # Convergence, bar, and energy-vs-quality plots
│   └── cli.py               # bbo-run / bbo-compare entry points
├── scripts/
│   ├── run_quick_demo.py    # Fast 5-algo demo with plots
│   ├── run_full_benchmark.py # Full cross-product benchmark
│   └── measure_energy.py    # Energy-focused comparison
├── tests/
│   ├── test_algorithms.py   # Algorithm correctness + convergence tests
│   ├── test_benchmarks.py   # Function values at known optima
│   └── test_metrics.py      # Metric computation unit tests
├── algorithms/              # Per-algorithm documentation (CMA-ES, DE, PSO)
├── benchmarks/              # Benchmark suite documentation (BBOB, CEC)
├── results/                 # Saved benchmark result CSVs + metadata
└── pyproject.toml
```

## Benchmark Functions

| Name | Group | Domain | f* |
|---|---|---|---|
| Sphere | separable | [-5.12, 5.12] | 0 |
| Step | separable | [-5.12, 5.12] | 0 |
| Sum of Powers | separable | [-1, 1] | 0 |
| Rosenbrock | unimodal | [-5, 10] | 0 |
| Ellipsoid | unimodal | [-5, 5] | 0 |
| Discus | unimodal | [-5, 5] | 0 |
| Bent Cigar | unimodal | [-5, 5] | 0 |
| Rastrigin | multimodal | [-5.12, 5.12] | 0 |
| Ackley | multimodal | [-32.768, 32.768] | 0 |
| Griewank | multimodal | [-600, 600] | 0 |
| Schwefel | multimodal | [-500, 500] | 0 |
| Levy | deceptive | [-10, 10] | 0 |
| Styblinski-Tang | deceptive | [-5, 5] | 0 |
| Michalewicz | deceptive | [0, π] | ~−0.966·d |

## Algorithms

| Algorithm | Class | Time per iter | Space |
|---|---|---|---|
| CMA-ES | `CMAESAlgorithm` | O(d² · λ) | O(d²) |
| Differential Evolution | `DEAlgorithm` | O(NP · d) | O(NP · d) |
| PSO | `PSOAlgorithm` | O(n · d) | O(n · d) |
| Random Search | `RandomSearch` | O(d) | O(d) |

## Example: Programmatic Usage

```python
from bbolib import BenchmarkRunner, CMAESAlgorithm, DEAlgorithm, PSOAlgorithm, get_function
from bbolib.plotting import plot_convergence

algos = [CMAESAlgorithm(), DEAlgorithm(), PSOAlgorithm()]
fn = get_function("rastrigin")

runner = BenchmarkRunner(measure_energy=True)
runner.run(algos, [fn], dimensions=[10, 20], budget=10_000, n_runs=10)

runner.print_summary("Rastrigin", dim=10, target=1e-6, budget=10_000)

results_by_algo = {a.name: runner.get_results(a.name, "Rastrigin", 10) for a in algos}
fig = plot_convergence(results_by_algo, budget=10_000, title="Rastrigin d=10")
fig.savefig("convergence.png")
```

## Energy Measurement

The `EnergyMonitor` context manager wraps any code block:

```python
from bbolib.energy import EnergyMonitor

with EnergyMonitor() as em:
    result = algo.minimize(fn, bounds, budget=10_000)

print(f"Energy: {em.report.best_estimate_joules:.2f} J")
print(f"Method: {em.report.measurement_method}")  # RAPL or psutil-estimate
```

On Linux with Intel/AMD CPUs, RAPL provides accurate CPU+DRAM readings. On Windows/macOS, a `psutil`-based CPU-time × TDP estimate is used as a fallback.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — adding algorithms, results data, or energy measurements is equally welcome.
