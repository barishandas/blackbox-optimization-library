"""
SPSA — Simultaneous Perturbation Stochastic Approximation (Spall 1992).

Estimates the gradient using only 2 function evaluations per step regardless
of dimension, by perturbing all parameters simultaneously with a Bernoulli
±1 random vector. Momentum smooths the noisy gradient estimates.
"""
import numpy as np
from .base import BBOAlgorithm, RunResult
from typing import Callable


class SPSAAlgorithm(BBOAlgorithm):
    """
    SPSA with Robbins-Monro decaying gain sequences and optional momentum.

    Parameters
    ----------
    a        : numerator of step-size gain a_k = a / (A + k + 1)^alpha
    c        : numerator of perturbation gain c_k = c / k^gamma
    A        : stability constant for step size
    alpha    : decay exponent for step size (standard: 0.602)
    gamma    : decay exponent for perturbation (standard: 0.101)
    momentum : exponential moving average coefficient for gradient (0 = no momentum)
               Smooths noisy gradient estimates; 0.9 is a good starting point.
    """

    name = "SPSA"

    def __init__(
        self,
        a: float = 0.1,
        c: float = 0.1,
        A: float = 100.0,
        alpha: float = 0.602,
        gamma: float = 0.101,
        momentum: float = 0.9,
    ):
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.momentum = momentum

    @property
    def dimension_complexity(self) -> str:
        return "O(d) per step — 2 evaluations regardless of d"

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
        d = bounds.shape[0]

        x = (lower + upper) / 2.0
        best_f = np.inf
        best_x = x.copy()
        history: list[float] = []
        evals = 0

        m = np.zeros(d)  # momentum buffer

        k = 1
        while evals + 2 <= budget:
            a_k = self.a / (self.A + k) ** self.alpha
            c_k = self.c / k ** self.gamma

            delta = rng.choice([-1.0, 1.0], size=d)

            x_plus  = np.clip(x + c_k * delta, lower, upper)
            x_minus = np.clip(x - c_k * delta, lower, upper)

            f_plus  = func(x_plus);  evals += 1
            f_minus = func(x_minus); evals += 1

            g_hat = (f_plus - f_minus) / (2.0 * c_k * delta)

            # apply momentum: exponential moving average of gradient estimates
            m = self.momentum * m + (1.0 - self.momentum) * g_hat

            x = np.clip(x - a_k * m, lower, upper)

            f_current = min(f_plus, f_minus)
            if f_current < best_f:
                best_f = f_current
                best_x = (x_plus if f_plus < f_minus else x_minus).copy()

            history.append(best_f)
            k += 1

        return RunResult(
            best_x=best_x,
            best_f=best_f,
            n_evaluations=evals,
            history_f=history,
        )
