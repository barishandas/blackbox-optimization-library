"""
FD — Finite Differences Gradient Descent.

Estimates the gradient via central differences (2d evaluations per step),
then takes a projected gradient step. Armijo backtracking line search adapts
the step size at each iteration rather than using a fixed decay schedule.
"""
import numpy as np
from .base import BBOAlgorithm, RunResult
from typing import Callable


class FDAlgorithm(BBOAlgorithm):
    """
    Central-differences gradient descent with Armijo backtracking line search.

    Parameters
    ----------
    step_size   : initial step size for line search
    h           : finite difference interval
    armijo_c    : Armijo sufficient-decrease constant (0 < c < 1)
    armijo_rho  : step size shrinkage factor per backtrack (0 < rho < 1)
    max_backtracks : maximum backtracking iterations per gradient step
    restarts    : number of random restarts when progress stalls
    """

    name = "FD"

    def __init__(
        self,
        step_size: float = 1.0,
        h: float = 1e-4,
        armijo_c: float = 1e-4,
        armijo_rho: float = 0.5,
        max_backtracks: int = 10,
        restarts: int = 4,
    ):
        self.step_size = step_size
        self.h = h
        self.armijo_c = armijo_c
        self.armijo_rho = armijo_rho
        self.max_backtracks = max_backtracks
        self.restarts = restarts

    @property
    def dimension_complexity(self) -> str:
        return "O(d) per step — 2d evaluations (central differences)"

    @property
    def space_complexity(self) -> str:
        return "O(d)"

    def _run_once(self, func, lower, upper, budget, rng, x0):
        d = lower.shape[0]
        x = x0.copy()
        f_x = func(x)
        best_f = f_x
        best_x = x.copy()
        history = [best_f]
        evals = 1
        step = self.step_size

        while evals + 2 * d + 1 <= budget:
            # central differences gradient
            grad = np.zeros(d)
            for i in range(d):
                e = np.zeros(d); e[i] = self.h
                f_p = func(np.clip(x + e, lower, upper)); evals += 1
                f_m = func(np.clip(x - e, lower, upper)); evals += 1
                grad[i] = (f_p - f_m) / (2.0 * self.h)

            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break  # at a stationary point — trigger restart
            grad_dir = grad / grad_norm

            # Armijo backtracking line search
            alpha = step
            for _ in range(self.max_backtracks):
                x_new = np.clip(x - alpha * grad_dir, lower, upper)
                f_new = func(x_new); evals += 1
                if f_new <= f_x - self.armijo_c * alpha * grad_norm:
                    break
                alpha *= self.armijo_rho

            x = x_new
            f_x = f_new
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()
            history.append(best_f)

            # expand step size slightly if line search accepted without shrinking
            step = min(alpha * 1.2, self.step_size)

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
            bx, bf, ev, hist = self._run_once(func, lower, upper, min(budget_per_restart, remaining), rng, x0)
            total_evals += ev
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
            running = best_f
            for h in hist:
                running = min(running, h)
                all_history.append(running)

        return RunResult(
            best_x=best_x,
            best_f=best_f,
            n_evaluations=total_evals,
            history_f=all_history,
        )
