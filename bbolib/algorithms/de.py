import numpy as np
from scipy.optimize import differential_evolution
from .base import BBOAlgorithm, RunResult
from typing import Callable


class DEAlgorithm(BBOAlgorithm):
    """
    Differential Evolution via scipy (Storn & Price 1997).

    Parameters
    ----------
    strategy    : DE variant string (default 'best1bin')
                  Options: 'best1bin', 'rand1bin', 'randtobest1bin',
                           'currenttobest1bin', 'best2bin', 'rand2bin'
    mutation    : F scale factor, float or (min, max) for dithering
    recombination : CR crossover rate
    popsize     : population size multiplier (actual size = popsize * d)
    """

    name = "DE"

    def __init__(
        self,
        strategy: str = "best1bin",
        mutation: float | tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        popsize: int = 15,
    ):
        self.strategy = strategy
        self.mutation = mutation
        self.recombination = recombination
        self.popsize = popsize

    @property
    def dimension_complexity(self) -> str:
        return "O(NP * d) per iteration"

    @property
    def space_complexity(self) -> str:
        return "O(NP * d)"

    def minimize(
        self,
        func: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        budget: int,
        seed: int = 0,
    ) -> RunResult:
        bounds = np.asarray(bounds)
        history: list[float] = []
        best_f = np.inf
        best_x = np.zeros(bounds.shape[0])
        evals_used = 0

        def tracked(x):
            nonlocal best_f, best_x, evals_used
            val = func(x)
            evals_used += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            history.append(best_f)
            return val

        try:
            differential_evolution(
                tracked,
                bounds=bounds.tolist(),
                strategy=self.strategy,
                mutation=self.mutation,
                recombination=self.recombination,
                popsize=self.popsize,
                maxiter=budget // max(1, self.popsize * bounds.shape[0]) + 1,
                seed=seed,
                tol=0,
                atol=0,
            )
        except Exception:
            pass

        return RunResult(
            best_x=best_x,
            best_f=best_f,
            n_evaluations=evals_used,
            history_f=history,
        )
