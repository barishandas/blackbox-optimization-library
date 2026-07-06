import numpy as np
from .base import BBOAlgorithm, RunResult
from typing import Callable


class RandomSearch(BBOAlgorithm):
    """Uniform random search baseline."""

    name = "Random Search"

    @property
    def dimension_complexity(self) -> str:
        return "O(d) per evaluation"

    @property
    def space_complexity(self) -> str:
        return "O(d)"

    def minimize(
        self,
        func: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        budget: int,
        seed: int = 0,
    ) -> RunResult:
        rng = np.random.default_rng(seed)
        bounds = np.asarray(bounds)
        lower, upper = bounds[:, 0], bounds[:, 1]

        best_f = np.inf
        best_x = (lower + upper) / 2.0
        history: list[float] = []

        for _ in range(budget):
            x = rng.uniform(lower, upper)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            history.append(best_f)

        return RunResult(
            best_x=best_x,
            best_f=best_f,
            n_evaluations=budget,
            history_f=history,
        )
