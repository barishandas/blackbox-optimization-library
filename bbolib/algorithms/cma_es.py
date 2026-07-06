import numpy as np
import cma
from .base import BBOAlgorithm, RunResult
from typing import Callable


class CMAESAlgorithm(BBOAlgorithm):
    """
    CMA-ES via the `cma` library (Hansen 2001).

    Parameters
    ----------
    sigma0      : initial step size (default 0.3 * search range)
    popsize     : override default population size (lambda)
    restarts    : number of IPOP restarts (0 = single run)
    """

    name = "CMA-ES"

    def __init__(self, sigma0: float | None = None, popsize: int | None = None, restarts: int = 5):
        self.sigma0 = sigma0
        self.popsize = popsize
        self.restarts = restarts

    @property
    def dimension_complexity(self) -> str:
        return "O(d^2 * lambda) per iteration"

    @property
    def space_complexity(self) -> str:
        return "O(d^2)"

    def minimize(
        self,
        func: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        budget: int,
        seed: int = 0,
    ) -> RunResult:
        bounds = np.asarray(bounds)
        d = bounds.shape[0]
        lower, upper = bounds[:, 0], bounds[:, 1]
        center = (lower + upper) / 2.0
        sigma0 = self.sigma0 if self.sigma0 is not None else 0.3 * (upper - lower).mean()

        history: list[float] = []
        best_f = np.inf
        best_x = center.copy()
        evals_used = 0

        def tracked(x):
            nonlocal best_f, best_x, evals_used
            val = func(np.asarray(x))
            evals_used += 1
            if val < best_f:
                best_f = val
                best_x = np.asarray(x).copy()
            history.append(best_f)
            return val

        opts = {
            "seed": seed,
            "maxfevals": budget,
            "bounds": [lower.tolist(), upper.tolist()],
            "verbose": -9,
        }
        if self.popsize:
            opts["popsize"] = self.popsize

        try:
            if self.restarts > 0:
                cma.fmin(tracked, center, sigma0, opts, restarts=self.restarts, bipop=True)
            else:
                cma.fmin(tracked, center, sigma0, opts)
        except cma.evolution_strategy.CMAEvolutionStrategyResult:
            pass
        except Exception:
            pass

        return RunResult(
            best_x=best_x,
            best_f=best_f,
            n_evaluations=evals_used,
            history_f=history,
        )
