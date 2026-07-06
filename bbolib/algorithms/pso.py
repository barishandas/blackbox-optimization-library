import numpy as np
from .base import BBOAlgorithm, RunResult
from typing import Callable


class PSOAlgorithm(BBOAlgorithm):
    """
    PSO with inertia weight and optional ring topology (Kennedy & Eberhart 1995).

    Parameters
    ----------
    n_particles : swarm size
    w           : inertia weight (or 'ldiw' for linearly decreasing)
    c1          : cognitive acceleration coefficient
    c2          : social acceleration coefficient
    w_end       : final inertia weight when w='ldiw'
    topology    : 'global' (canonical gbest) or 'ring' (local lbest, k=2 neighbours)
                  Ring topology dramatically reduces premature convergence on
                  multimodal functions by limiting information propagation.
    """

    name = "PSO"

    def __init__(
        self,
        n_particles: int = 30,
        w: float | str = "ldiw",
        c1: float = 2.05,
        c2: float = 2.05,
        w_end: float = 0.4,
        topology: str = "ring",
    ):
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.w_end = w_end
        self.topology = topology

    @property
    def dimension_complexity(self) -> str:
        return "O(n_particles * d) per iteration"

    @property
    def space_complexity(self) -> str:
        return "O(n_particles * d)"

    def _neighbourhood_best(self, pbest_f: np.ndarray, pbest_pos: np.ndarray) -> np.ndarray:
        """Return the best position in each particle's ring neighbourhood (k=2 neighbours each side)."""
        n = self.n_particles
        nbest = np.empty_like(pbest_pos)
        for i in range(n):
            neighbours = [(i - 2) % n, (i - 1) % n, i, (i + 1) % n, (i + 2) % n]
            best_idx = neighbours[int(np.argmin(pbest_f[neighbours]))]
            nbest[i] = pbest_pos[best_idx]
        return nbest

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

        pos = rng.uniform(lower, upper, size=(self.n_particles, d))
        vel = rng.uniform(-span, span, size=(self.n_particles, d)) * 0.1
        pbest_pos = pos.copy()
        pbest_f = np.full(self.n_particles, np.inf)
        gbest_pos = pos[0].copy()
        gbest_f = np.inf

        history: list[float] = []
        evals_used = 0

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

            # pick attractor: global best or local neighbourhood best
            if self.topology == "ring":
                attractor = self._neighbourhood_best(pbest_f, pbest_pos)
            else:
                attractor = np.tile(gbest_pos, (self.n_particles, 1))

            r1 = rng.uniform(0, 1, size=(self.n_particles, d))
            r2 = rng.uniform(0, 1, size=(self.n_particles, d))

            vel = (
                w_t * vel
                + self.c1 * r1 * (pbest_pos - pos)
                + self.c2 * r2 * (attractor - pos)
            )
            v_max = 0.2 * span
            vel = np.clip(vel, -v_max, v_max)

            pos = np.clip(pos + vel, lower, upper)

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
