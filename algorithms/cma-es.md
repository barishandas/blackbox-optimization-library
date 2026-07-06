# CMA-ES — Covariance Matrix Adaptation Evolution Strategy

> Introduced by Hansen & Ostermeier, 1996. DOI: [10.1162/106365601750190398](https://doi.org/10.1162/106365601750190398)

## Summary

CMA-ES is a stochastic, derivative-free optimizer for continuous domains. It maintains a multivariate Gaussian distribution over the search space and iteratively updates its mean and full covariance matrix based on successful candidate solutions. The covariance adaptation allows it to learn dependencies between variables and rescale along favorable directions, making it highly effective on ill-conditioned and non-separable problems.

## Parameters

| Parameter | Typical Value | Description |
|---|---|---|
| `lambda` (population size) | `4 + floor(3 * ln(d))` | Number of offspring per generation |
| `mu` (parents) | `lambda / 2` | Number of top candidates used for update |
| `sigma` (step size) | problem-dependent | Initial standard deviation (step size) |
| `c_sigma`, `d_sigma` | auto | Step-size control constants |
| `c_c`, `c_1`, `c_mu` | auto | Covariance update learning rates |

All adaptation parameters have well-established default formulas based on dimension `d` — CMA-ES generally requires only `sigma` to be set by the user.

## Computational Complexity

| Aspect | Complexity |
|---|---|
| Time per iteration | O(d^2 * lambda) |
| Space | O(d^2) — for storing the covariance matrix |
| Evaluations to convergence (typical) | O(d) to O(d^2) depending on conditioning |

The O(d^2) memory and per-iteration cost makes standard CMA-ES impractical above ~1000 dimensions. Variants like sep-CMA-ES (diagonal covariance) reduce this to O(d).

## Energy Efficiency

| Hardware | Problem | Evaluations | Energy (J) | Quality Reached |
|---|---|---|---|---|
| — | — | — | — | — |

*Contributions welcome — use the methodology described in [CONTRIBUTING.md](../CONTRIBUTING.md).*

## Convergence Performance

| Benchmark Suite | Dimension | Budget (FEs) | Success Rate | Median FEs to Target |
|---|---|---|---|---|
| BBOB (f1–f24) | 5 | 10,000 | ~90% | varies by function |
| BBOB (f1–f24) | 20 | 100,000 | ~80% | varies by function |

CMA-ES is considered state-of-the-art on the BBOB benchmark for moderate dimensions (2–40). Performance degrades on multimodal functions with many local optima without restart strategies.

## Strengths and Weaknesses

**Strengths:**
- Invariant to rotation and scaling of the search space
- Self-adaptive — minimal hyperparameter tuning required
- Excellent on smooth, unimodal, and moderately multimodal functions
- Well-studied theoretical guarantees on convergence rates

**Weaknesses:**
- O(d^2) cost limits scalability beyond ~1000 dimensions
- Can struggle on highly multimodal landscapes without restarts (e.g., IPOP-CMA-ES)
- Not natively suited for discrete or mixed-type search spaces

## Recommended Use Cases

- Continuous black-box optimization in low-to-moderate dimensions (d ≤ 500)
- Hyperparameter tuning of ML models with continuous parameters
- Engineering design problems (aerodynamic shape optimization, controller tuning)
- As a baseline or gold-standard comparator in BBO benchmarking studies

## References

1. Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*, 9(2), 159–195.
2. Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial. [arXiv:1604.00772](https://arxiv.org/abs/1604.00772)
3. Hansen, N., et al. (2010). Comparing results of 31 algorithms from the black-box optimization benchmarking BBOB-2009. *GECCO Companion*.
