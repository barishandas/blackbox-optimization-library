# FD — Finite Differences Gradient Descent

> Classical numerical analysis method; modern BBO context: Nesterov & Spokoiny (2017).

## Summary

Finite Differences (FD) estimates the gradient by evaluating the function at small perturbations of each coordinate separately, then performs gradient descent. Using central differences, each gradient estimate requires **2d function evaluations** (one forward and one backward perturbation per dimension). This is the most direct way to approximate gradient-based optimization without access to analytic derivatives.

In the BBO context, FD serves as a reference point: it represents what gradient information can achieve when it must be numerically estimated, at the cost of 2d evaluations per step.

## Parameters

| Parameter | Typical Value | Description |
|---|---|---|
| `step_size` | 0.1–1.0 | Initial gradient descent step length |
| `h` | 1e-4 to 1e-6 | Finite difference interval |
| `decay` | 0.999 | Multiplicative step size decay per iteration |
| `min_step` | 1e-8 | Minimum step size floor |
| `restarts` | 3 | Random restarts when progress stalls |

`h` should be chosen relative to the function's scale — too large introduces truncation error, too small causes cancellation error.

## Computational Complexity

| Aspect | Complexity |
|---|---|
| Time per step | O(d) — exactly 2d evaluations (central differences) |
| Space | O(d) |
| Evaluations to convergence | O(d / ε) for strongly convex functions |

FD is 2d× more expensive per step than SPSA but provides a lower-variance gradient estimate. On a budget of B evaluations, FD takes B/(2d) gradient steps.

## Energy Efficiency

| Hardware | Problem | Evaluations | Energy (J) | Quality Reached |
|---|---|---|---|---|
| — | — | — | — | — |

*Contributions welcome.*

## Convergence Performance

| Benchmark Suite | Dimension | Budget (FEs) | Success Rate | Median FEs to Target |
|---|---|---|---|---|
| Sphere | 10 | 50,000 | high | — |
| Rosenbrock | 10 | 50,000 | moderate | — |
| Rastrigin | 10 | 50,000 | low | — |

FD performs well on smooth unimodal functions but is impractical for d > 100 at typical budgets (each gradient step costs 20+ evaluations at d=10, 200+ at d=100).

## Strengths and Weaknesses

**Strengths:**
- Low-variance gradient estimates compared to SPSA
- Deterministic gradient direction (for fixed h)
- Simple and transparent — easy to debug and understand
- Effective on smooth, low-dimensional problems

**Weaknesses:**
- Cost grows linearly with dimension — impractical for d > 100 at typical budgets
- Trapped by local optima on multimodal functions; restarts help but don't solve it
- h requires tuning — wrong scale causes numerical errors
- Not rotation-invariant; struggles on ill-conditioned problems without preconditioning

## Recommended Use Cases

- Low-dimensional smooth optimization (d ≤ 50) with expensive function evaluations
- Benchmarking reference: shows what gradient information achieves at the cost of 2d evaluations/step
- Problems where analytic gradients are unavailable but the function is smooth
- Comparison baseline against SPSA (shows value of per-coordinate gradient info)

## References

1. Nesterov, Y., & Spokoiny, V. (2017). Random gradient-free minimization of convex functions. *Foundations of Computational Mathematics*, 17(2), 527–566.
2. Conn, A. R., Scheinberg, K., & Vicente, L. N. (2009). *Introduction to Derivative-Free Optimization*. SIAM.
3. Larson, J., Menickelly, M., & Wild, S. M. (2019). Derivative-free optimization methods. *Acta Numerica*, 28, 287–404.
