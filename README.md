# Black-Box Optimization Algorithm Library

A continuously updated reference library tracking **energy efficiency**, **computational complexity**, and **convergence performance** of black-box optimization (BBO) algorithms.

## What This Is

Black-box optimization algorithms operate without access to gradient information — they treat the objective function as a black box and evaluate it iteratively. This library benchmarks and documents their performance across three key dimensions:

| Dimension | What We Measure |
|---|---|
| **Energy Efficiency** | Energy consumed per function evaluation or per unit of solution quality |
| **Computational Complexity** | Time and space complexity; wall-clock scaling with problem dimension |
| **Convergence** | Speed to reach target quality; success rate; anytime performance curves |

## Algorithms Covered

### Evolutionary / Population-Based
- [CMA-ES](algorithms/cma-es.md) — Covariance Matrix Adaptation Evolution Strategy
- [DE](algorithms/de.md) — Differential Evolution
- [GA](algorithms/ga.md) — Genetic Algorithm

### Swarm Intelligence
- [PSO](algorithms/pso.md) — Particle Swarm Optimization
- [ABC](algorithms/abc.md) — Artificial Bee Colony

### Bayesian / Surrogate-Based
- [GP-BO](algorithms/gp-bo.md) — Gaussian Process Bayesian Optimization
- [SMAC](algorithms/smac.md) — Sequential Model-based Algorithm Configuration

### Direct Search
- [Nelder-Mead](algorithms/nelder-mead.md) — Simplex Method
- [DIRECT](algorithms/direct.md) — Dividing Rectangles

### Random / Baseline
- [Random Search](algorithms/random-search.md)

## Benchmark Suites

Algorithms are evaluated against standard test suites:

- **BBOB** (Black-Box Optimization Benchmarking, COCO framework) — 24 noiseless functions
- **CEC** (IEEE Congress on Evolutionary Computation) suites
- **Real-world problems** — engineering design, hyperparameter tuning, chemical optimization

See [`benchmarks/`](benchmarks/) for test function details and experimental setups.

## Results

Raw and aggregated results are stored in [`results/`](results/), organized by algorithm and benchmark suite. Each result entry records:

- Hardware/environment used
- Number of function evaluations
- Best solution found
- Convergence curve data
- Energy readings (where available)

## Contributing

To add a new algorithm or update an existing entry, use the [algorithm template](algorithms/_template.md) and open a PR. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Repository Structure

```
blackbox-optimization-library/
├── algorithms/          # Per-algorithm documentation and metadata
├── benchmarks/          # Test function definitions and experimental setups
├── results/             # Raw and summarized benchmark results
└── tools/               # Scripts for running benchmarks and generating plots
```
