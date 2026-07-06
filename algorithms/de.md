# DE — Differential Evolution

> Introduced by Storn & Price, 1997. DOI: [10.1023/A:1008202821328](https://doi.org/10.1023/A:1008202821328)

## Summary

Differential Evolution is a population-based optimizer that generates trial vectors by adding the weighted difference between two randomly selected population members to a third (mutation), then mixing the result with the target vector (crossover). Only if the trial vector is better does it replace the target — selection is purely greedy. DE is simple, robust, and requires very few parameter choices.

## Parameters

| Parameter | Typical Value | Description |
|---|---|---|
| `NP` (population size) | `5d` to `10d` | Number of individuals |
| `F` (scale factor) | 0.4–1.0 | Differential mutation weight |
| `CR` (crossover rate) | 0.1–1.0 | Probability of taking trial vector component |
| Strategy | `DE/rand/1/bin` | Mutation + crossover variant |

Common strategies: `DE/rand/1/bin`, `DE/best/1/bin`, `DE/current-to-best/1/bin`.

## Computational Complexity

| Aspect | Complexity |
|---|---|
| Time per iteration | O(NP * d) |
| Space | O(NP * d) |
| Evaluations to convergence (typical) | O(d * NP) to O(d^2) |

Scales linearly with dimension for fixed population ratio NP/d.

## Energy Efficiency

| Hardware | Problem | Evaluations | Energy (J) | Quality Reached |
|---|---|---|---|---|
| — | — | — | — | — |

*Contributions welcome.*

## Convergence Performance

| Benchmark Suite | Dimension | Budget (FEs) | Success Rate | Median FEs to Target |
|---|---|---|---|---|
| BBOB (f1–f24) | 10 | 50,000 | good on separable | varies |
| CEC 2017 | 30 | 300,000 | competitive | — |

DE is a consistent top performer on CEC benchmarks and is often used as a strong baseline. Adaptive variants (jDE, SHADE, L-SHADE) significantly improve robustness by auto-tuning F and CR.

## Strengths and Weaknesses

**Strengths:**
- Extremely simple implementation
- Few parameters; `F=0.8, CR=0.9` often works well out of the box
- Strong on separable and partially separable problems
- Excellent anytime performance curve — finds good solutions early

**Weaknesses:**
- Slower on non-separable, ill-conditioned problems than CMA-ES
- Fixed population can lead to premature convergence on deceptive landscapes
- Standard form is not rotation-invariant

## Recommended Use Cases

- General-purpose global optimization, especially when simplicity matters
- Problems where a strong baseline is needed for comparison
- Hyperparameter tuning and neural architecture search
- Real-parameter optimization in engineering and scientific computing

## References

1. Storn, R., & Price, K. (1997). Differential evolution — a simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341–359.
2. Das, S., & Mallipeddi, R. (2011). Differential evolution: A survey of the state-of-the-art. *IEEE TEC*, 15(1), 4–31.
3. Tanabe, R., & Fukunaga, A. (2013). Success-history based parameter adaptation for differential evolution. *IEEE CEC*.
