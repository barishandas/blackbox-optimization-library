"""
FD — Finite Differences Gradient Descent.

Estimates the gradient using central differences (2d evaluations per step),
then takes a projected gradient descent step. Useful as a reference for
how well gradient information helps when it can be numerically estimated.
"""
import numpy as np
from .base import BBOAlgorithm, RunResult
from typing import Callable


class FDAlgorithm(BBOAlgorithm):
    """
    Central-differences gradient descent with decaying step size.

    Parameters
    ----------
    step_size   : initial gradient descent step size
    h           : finite difference interval (perturbation size)
    decay       : multiplicative step size decay per iteration
    min_step    : lower bound on step size
    restarts    : number of random restarts when progress stalls
    """

    name = "FD"

    def __init__(
        self,
        step_size: float = 0.5,
        h: float = 1e-4,
        decay: float = 0.999,
        min_step: float = 1e-8,
        restarts: int = 3,
    ):
        self.step_size = step_size
        self.h = h
        self.decay = decay
        self.min_step = min_step
        self.restarts = restarts

    @property
    def dimension_complexity(self) -> str:
        return "O(d) per step — 2d evaluations (central differences)"

    @property
    def space_complexity(self) -> str:
        return "O(d)"

    def _run_once(self, func, lower, upper, budget, rng, x0=None):
        d = lower.shape[0]
        x = x0 if x0 is not None else rng.uniform(lower, upper)
        step = self.step_size
        best_f = func(x)
        best_x = x.copy()
        history = [best_f]
        evals = 1

        while evals + 2 * d <= budget:
            grad = np.zeros(d)
            for i in range(d):
                e = np.zeros(d); e[i] = self.h
                f_plus  = func(np.clip(x + e, lower, upper)); evals += 1
                f_minus = func(np.clip(x - e, lower, upper)); evals += 1
                grad[i] = (f_plus - f_minus) / (2.0 * self.h)

            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                grad = grad / grad_norm  # normalise to unit step

            x_new = np.clip(x - step * grad, lower, upper)
            f_new = func(x_new); evals += 1

            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

            history.append(best_f)
            x = x_new
            step = max(step * self.decay, self.min_step)

        return best_x, best_f, evals, history

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
        all_history: list[float] = []
        total_evals = 0

        budget_per_restart = budget // (self.restarts + 1)

        for r in range(self.restarts + 1):
            remaining = budget - total_evals
            if remaining <= 0:
                break
            x0 = (lower + upper) / 2.0 if r == 0 else rng.uniform(lower, upper)
            bx, bf, evals, hist = self._run_once(func, lower, upper, min(budget_per_restart, remaining), rng, x0)
            total_evals += evals
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
            # carry forward the running best across restarts
            running_best = best_f
            for h in hist:
                running_best = min(running_best, h)
                all_history.append(running_best)

        return RunResult(
            best_x=best_x,
            best_f=best_f,
            n_evaluations=total_evals,
            history_f=all_history,
        )
