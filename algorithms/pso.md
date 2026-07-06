# PSO — Particle Swarm Optimization

> Introduced by Kennedy & Eberhart, 1995. DOI: [10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968)

## Summary

PSO simulates the social behavior of birds flocking or fish schooling. A swarm of particles moves through the search space, each tracking its personal best position and the global best position found by any particle. Velocity updates blend inertia, attraction toward the personal best, and attraction toward the global best, producing emergent collective search behavior.

## Parameters

| Parameter | Typical Value | Description |
|---|---|---|
| `n_particles` | 20–50 | Swarm size |
| `w` (inertia weight) | 0.4–0.9 (often decaying) | Momentum term dampening velocity |
| `c1` (cognitive) | ~2.0 | Pull toward personal best |
| `c2` (social) | ~2.0 | Pull toward global best |
| `v_max` | ~10–20% of search range | Velocity clamp per dimension |

## Computational Complexity

| Aspect | Complexity |
|---|---|
| Time per iteration | O(n_particles * d) |
| Space | O(n_particles * d) |
| Evaluations to convergence (typical) | O(d) to O(d * n_particles) |

Linear in both population size and dimension — PSO scales well to high-dimensional problems compared to CMA-ES, though convergence quality may suffer.

## Energy Efficiency

| Hardware | Problem | Evaluations | Energy (J) | Quality Reached |
|---|---|---|---|---|
| — | — | — | — | — |

*Contributions welcome.*

## Convergence Performance

| Benchmark Suite | Dimension | Budget (FEs) | Success Rate | Median FEs to Target |
|---|---|---|---|---|
| BBOB (selected) | 10 | 50,000 | moderate | varies |
| CEC 2017 | 30 | 300,000 | varies | — |

PSO converges quickly on unimodal problems but is prone to premature convergence on multimodal landscapes. Variants like SPSO-2011 and CLPSO improve multimodal performance.

## Strengths and Weaknesses

**Strengths:**
- Simple to implement, few hyperparameters
- Good early-phase exploration
- Linear time and space complexity — scales to high dimensions
- Effective on many unimodal and lightly multimodal problems

**Weaknesses:**
- Prone to premature convergence (swarm collapses to a local optimum)
- Parameter sensitivity — inertia weight tuning is important
- No formal convergence guarantees in general

## Recommended Use Cases

- Large-scale continuous optimization where O(d^2) methods are too costly
- Problems where fast approximate solutions are more valuable than precision
- Engineering applications: antenna design, power systems, scheduling

## References

1. Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. *Proceedings of ICNN*, 4, 1942–1948.
2. Clerc, M., & Kennedy, J. (2002). The particle swarm — explosion, stability, and convergence in a multidimensional complex space. *IEEE TEC*, 6(1), 58–73.
3. Zambrano-Bigiarini, M., et al. (2013). Standard Particle Swarm Optimisation 2011 at CEC-2013. *IEEE CEC*.
