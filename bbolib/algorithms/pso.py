import numpy as np
from .base import BBOAlgorithm, RunResult
from typing import Callable


class PSOAlgorithm(BBOAlgorithm):
    """
    Canonical PSO with inertia weight (Kennedy & Eberhart 1995).

    Parameters
    ----------
    n_particles : swarm size
    w           : inertia weight (or 'ldiw' for linearly decreasing)
    c1          : cognitive acceleration coefficient
    c2          : social acceleration coefficient
    w_end       : final inertia weight when w='ldiw'
    """

    name = "PSO"

    def __init__(
        self,
        n_particles: int = 30,
        w: float | str = "ldiw",
        c1: float = 2.05,
        c2: float = 2.05,
        w_end: float = 0.4,
    ):
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_end = w_end

    @property
    def dimension_complexity(self) -> str:
        return "O(n_particles * d) per iteration"

    @property
    def space_complexity(self) -> str:
        return "O(n_particles * d)"

    def minimize(
        self,
        func: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        budget: int,
        seed: int = 0,
    ) -> RunResult:
        rng = np.random.default_rng(seed)
        bounds = np.asarray(bounds)
        d = bounds.shape[0]
        lower, upper = bounds[:, 0], bounds[:, 1]
        span = upper - lower

        # initialise swarm
        pos = rng.uniform(lower, upper, size=(self.n_particles, d))
        vel = rng.uniform(-span, span, size=(self.n_particles, d)) * 0.1
        pbest_pos = pos.copy()
        pbest_f = np.full(self.n_particles, np.inf)
        gbest_pos = pos[0].copy()
        gbest_f = np.inf

        history: list[float] = []
        evals_used = 0

        # evaluate initial population
        for i in range(self.n_particles):
            if evals_used >= budget:
                break
            f = func(pos[i])
            evals_used += 1
            pbest_f[i] = f
            if f < gbest_f:
                gbest_f = f
                gbest_pos = pos[i].copy()
            history.append(gbest_f)

        max_iter = (budget - self.n_particles) // self.n_particles + 1
        w_start = 0.9 if self.w == "ldiw" else float(self.w)

        for t in range(max_iter):
            if evals_used >= budget:
                break
            w_t = (w_start - self.w_end) * (1 - t / max_iter) + self.w_end if self.w == "ldiw" else w_start

            r1 = rng.uniform(0, 1, size=(self.n_particles, d))
            r2 = rng.uniform(0, 1, size=(self.n_particles, d))

            vel = (
                w_t * vel
                + self.c1 * r1 * (pbest_pos - pos)
                + self.c2 * r2 * (gbest_pos - pos)
            )
            # velocity clamp
            v_max = 0.2 * span
            vel = np.clip(vel, -v_max, v_max)

            pos = pos + vel
            pos = np.clip(pos, lower, upper)

            for i in range(self.n_particles):
                if evals_used >= budget:
                    break
                f = func(pos[i])
                evals_used += 1
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_pos[i] = pos[i].copy()
                if f < gbest_f:
                    gbest_f = f
                    gbest_pos = pos[i].copy()
                history.append(gbest_f)

        return RunResult(
            best_x=gbest_pos,
            best_f=gbest_f,
            n_evaluations=evals_used,
            history_f=history,
        )
