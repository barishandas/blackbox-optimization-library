# PEPG / PGPE — Parameter-Exploring Policy Gradients

> Introduced by Sehnke et al., 2010. DOI: [10.1016/j.neunet.2009.12.004](https://doi.org/10.1016/j.neunet.2009.12.004)

## Summary

PEPG (also known as PGPE — Policy Gradients with Parameter-based Exploration) maintains a Gaussian search distribution N(μ, σ) over the parameter space and updates both the mean μ and per-dimension standard deviations σ using likelihood-ratio / policy gradient estimates.

The key insight is **antithetic (symmetric) sampling**: for each noise vector ε, it evaluates both x⁺ = μ + σε and x⁻ = μ − σε. The difference in their fitness values directly estimates the gradient of μ, while the sum estimates the gradient of σ. This halves variance compared to single-sided sampling.

PEPG lies at the intersection of evolution strategies and policy gradient methods. It is closely related to NES (Natural Evolution Strategies) but uses a simpler, non-natural gradient update.

## Parameters

| Parameter | Typical Value | Description |
|---|---|---|
| `n_samples` | 10–50 | Antithetic pairs per generation (population = 2×n_samples) |
| `lr_mu` | 0.05–0.2 | Learning rate for mean μ |
| `lr_sigma` | 0.01–0.1 | Learning rate for std dev σ |
| `sigma_init` | 0.1–1.0 × search range | Initial exploration spread |
| `sigma_min` | 1e-5 | Minimum σ to prevent collapse |
| `baseline` | `'mean'` | Reward baseline for variance reduction |

## Computational Complexity

| Aspect | Complexity |
|---|---|
| Time per generation | O(n_samples × d) — 2×n_samples evaluations |
| Space | O(n_samples × d) |
| Evaluations to convergence | O(d × n_samples / ε) typically |

Per-generation cost is similar to PSO but the search distribution is maintained explicitly, allowing principled σ adaptation.

## Energy Efficiency

| Hardware | Problem | Evaluations | Energy (J) | Quality Reached |
|---|---|---|---|---|
| — | — | — | — | — |

*Contributions welcome.*

## Convergence Performance

| Benchmark Suite | Dimension | Budget (FEs) | Success Rate | Median FEs to Target |
|---|---|---|---|---|
| Sphere | 10 | 50,000 | high | — |
| Rosenbrock | 10 | 50,000 | moderate–high | — |
| Rastrigin | 10 | 50,000 | low | — |

PEPG performs well on unimodal problems and handles moderate ill-conditioning via σ adaptation. It underperforms CMA-ES on non-separable problems because it adapts per-dimension variances but not covariances (no full covariance matrix).

## Strengths and Weaknesses

**Strengths:**
- Principled probabilistic framework — updates are gradient estimates of the search distribution
- Antithetic sampling reduces variance efficiently
- Per-dimension σ adapts automatically — handles axis-aligned anisotropy
- Strong connection to policy gradient theory — directly applicable to RL
- Lower memory than CMA-ES (O(d) vs O(d²))

**Weaknesses:**
- No full covariance adaptation — cannot rotate to align with problem structure
- Learning rates require tuning; less self-contained than CMA-ES
- Underperforms CMA-ES on non-separable, ill-conditioned problems
- Can collapse σ → 0 prematurely if learning rate is too high

## Recommended Use Cases

- Reinforcement learning parameter optimization (original application)
- High-dimensional optimization where CMA-ES O(d²) cost is prohibitive
- Neuroevolution: optimizing neural network weights
- Problems with separable or nearly-separable structure

## References

1. Sehnke, F., et al. (2010). Parameter-exploring policy gradients. *Neural Networks*, 23(4), 551–559.
2. Wierstra, D., et al. (2014). Natural evolution strategies. *JMLR*, 15(1), 949–980.
3. Salimans, T., et al. (2017). Evolution strategies as a scalable alternative to reinforcement learning. [arXiv:1703.03864](https://arxiv.org/abs/1703.03864)
