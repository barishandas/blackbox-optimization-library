# SPSA — Simultaneous Perturbation Stochastic Approximation

> Introduced by Spall, 1992. DOI: [10.1109/9.119632](https://doi.org/10.1109/9.119632)

## Summary

SPSA estimates the gradient using only **2 function evaluations per step**, regardless of dimension. Instead of perturbing one coordinate at a time (as finite differences does), it perturbs all coordinates simultaneously using a random Bernoulli ±1 vector. The resulting estimate is a noisy but unbiased approximation of the gradient direction. Step sizes and perturbation sizes follow Robbins-Monro decaying sequences to ensure convergence.

This makes SPSA particularly attractive when evaluations are expensive and dimension is large — the cost per step is O(1) in the number of evaluations rather than O(d).

## Parameters

| Parameter | Typical Value | Description |
|---|---|---|
| `a` | 0.01–0.5 | Numerator of step-size gain sequence |
| `c` | 0.01–0.5 | Numerator of perturbation gain sequence |
| `A` | ~10% of max iterations | Stability constant |
| `alpha` | 0.602 | Step-size decay exponent (standard value) |
| `gamma` | 0.101 | Perturbation decay exponent (standard value) |

The standard values α=0.602, γ=0.101 satisfy Robbins-Monro conditions. Tuning `a` and `c` to the problem scale is often the most impactful choice.

## Computational Complexity

| Aspect | Complexity |
|---|---|
| Time per step | O(d) — 2 function evaluations |
| Space | O(d) |
| Evaluations to convergence | O(1/ε²) in theory; problem-dependent in practice |

SPSA uses exactly 2 evaluations per iteration regardless of d, making it far cheaper per step than finite differences (which needs 2d evaluations).

## Energy Efficiency

| Hardware | Problem | Evaluations | Energy (J) | Quality Reached |
|---|---|---|---|---|
| — | — | — | — | — |

*Contributions welcome.*

## Convergence Performance

| Benchmark Suite | Dimension | Budget (FEs) | Success Rate | Median FEs to Target |
|---|---|---|---|---|
| Sphere | 10 | 50,000 | moderate | — |
| Rosenbrock | 10 | 50,000 | low–moderate | — |

SPSA converges reliably on smooth unimodal functions. On multimodal functions it typically gets trapped in local optima. Convergence is slower than CMA-ES on well-conditioned problems but scales far better with dimension.

## Strengths and Weaknesses

**Strengths:**
- Only 2 function evaluations per step — extremely evaluation-efficient per iteration
- Scales to very high dimensions (d > 10,000) where population methods are impractical
- Simple to implement; few hyperparameters
- Theoretical convergence guarantees under standard conditions

**Weaknesses:**
- Noisy gradient estimates — slow convergence on ill-conditioned or multimodal problems
- Sensitive to gain sequence parameters (`a`, `c`) — requires tuning per problem
- Not rotation-invariant; no covariance adaptation
- Easily trapped in local optima

## Recommended Use Cases

- High-dimensional smooth optimization (d > 500) where population methods are too costly
- Stochastic/noisy objectives where gradient estimates are inherently noisy
- Online and streaming optimization problems
- Reinforcement learning policy gradient estimation

## References

1. Spall, J. C. (1992). Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. *IEEE TAC*, 37(3), 332–341.
2. Spall, J. C. (1998). An overview of the simultaneous perturbation method for efficient optimization. *Johns Hopkins APL Technical Digest*, 19(4), 482–492.
3. Spall, J. C. (2003). *Introduction to Stochastic Search and Optimization*. Wiley.
