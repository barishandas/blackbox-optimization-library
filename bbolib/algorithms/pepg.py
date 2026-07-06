"""
PEPG / PGPE — Parameter-Exploring Policy Gradients (Sehnke et al. 2010).

Maintains a Gaussian search distribution N(μ, σ) over parameters.
Uses antithetic (symmetric) sampling to reduce variance: for each noise
vector ε, evaluates both μ+σε and μ-σε and uses the difference to
estimate the natural gradient of μ and σ simultaneously.
"""
import numpy as np
from .base import BBOAlgorithm, RunResult
from typing import Callable


class PEPGAlgorithm(BBOAlgorithm):
    """
    PEPG with antithetic sampling and per-dimension step sizes.

    Parameters
    ----------
    n_samples   : number of antithetic pairs per generation (population = 2*n_samples)
    lr_mu       : learning rate for mean μ
    lr_sigma    : learning rate for standard deviation σ
    sigma_init  : initial σ (per dimension)
    sigma_min   : minimum σ (prevents collapse)
    baseline    : 'mean' uses average reward as baseline to reduce variance
    """

    name = "PEPG"

    def __init__(
        self,
        n_samples: int = 20,
        lr_mu: float = 0.1,
        lr_sigma: float = 0.05,
        sigma_init: float = 0.5,
        sigma_min: float = 1e-5,
        baseline: str = "mean",
    ):
        self.n_samples = n_samples
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.baseline = baseline

    @property
    def dimension_complexity(self) -> str:
        return "O(n_samples * d) per generation"

    @property
    def space_complexity(self) -> str:
        return "O(n_samples * d)"

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
        span = upper - lower

        mu = (lower + upper) / 2.0
        sigma = np.full(d, self.sigma_init * span.mean())

        best_f = np.inf
        best_x = mu.copy()
        history: list[float] = []
        evals = 0

        while evals + 2 * self.n_samples <= budget:
            epsilons = rng.standard_normal((self.n_samples, d))

            f_plus  = np.zeros(self.n_samples)
            f_minus = np.zeros(self.n_samples)

            for i in range(self.n_samples):
                x_plus  = np.clip(mu + sigma * epsilons[i], lower, upper)
                x_minus = np.clip(mu - sigma * epsilons[i], lower, upper)
                f_plus[i]  = func(x_plus);  evals += 1
                f_minus[i] = func(x_minus); evals += 1

                for f_val, x_val in [(f_plus[i], x_plus), (f_minus[i], x_minus)]:
                    if f_val < best_f:
                        best_f = f_val
                        best_x = x_val.copy()
                history.append(best_f)
                history.append(best_f)

            # rewards (negated for minimisation)
            r_plus  = -f_plus
            r_minus = -f_minus

            if self.baseline == "mean":
                b = (r_plus + r_minus).mean()
            else:
                b = 0.0

            # μ gradient: Σ [(r+ - r-) / 2] * ε
            mu_grad = np.zeros(d)
            sigma_grad = np.zeros(d)
            for i in range(self.n_samples):
                advantage = (r_plus[i] - r_minus[i]) / 2.0
                mu_grad += advantage * epsilons[i]

                # σ gradient: uses symmetric baseline (r+ + r-)/2 - b
                sym_reward = (r_plus[i] + r_minus[i]) / 2.0 - b
                sigma_grad += sym_reward * (epsilons[i] ** 2 - 1.0)

            mu_grad    /= self.n_samples
            sigma_grad /= self.n_samples

            mu    = np.clip(mu    + self.lr_mu    * mu_grad,    lower, upper)
            sigma = np.maximum(sigma + self.lr_sigma * sigma_grad / sigma, self.sigma_min)

        return RunResult(
            best_x=best_x,
            best_f=best_f,
            n_evaluations=evals,
            history_f=history,
        )
