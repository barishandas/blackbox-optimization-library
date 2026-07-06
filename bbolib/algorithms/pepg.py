"""
PEPG / PGPE — Parameter-Exploring Policy Gradients (Sehnke et al. 2010).

Maintains a Gaussian search distribution N(μ, σ) and updates both using
antithetic sampling. Rank-based fitness shaping reduces the influence of
extreme outliers and improves stability on multimodal landscapes.
"""
import numpy as np
from .base import BBOAlgorithm, RunResult
from typing import Callable


class PEPGAlgorithm(BBOAlgorithm):
    """
    PEPG with antithetic sampling, per-dimension σ, and rank-based fitness shaping.

    Parameters
    ----------
    n_samples       : antithetic pairs per generation (population = 2 * n_samples)
    lr_mu           : learning rate for mean μ
    lr_sigma        : learning rate for std dev σ
    sigma_init      : initial σ as fraction of search range
    sigma_min       : minimum σ floor to prevent collapse
    fitness_shaping : if True, replace raw rewards with centred ranks before
                      computing gradients — reduces outlier influence and
                      improves convergence on noisy/multimodal problems
    """

    name = "PEPG"

    def __init__(
        self,
        n_samples: int = 20,
        lr_mu: float = 0.1,
        lr_sigma: float = 0.05,
        sigma_init: float = 0.3,
        sigma_min: float = 1e-5,
        fitness_shaping: bool = True,
    ):
        self.n_samples = n_samples
        self.lr_mu = lr_mu
        self.lr_sigma = lr_sigma
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.fitness_shaping = fitness_shaping

    @property
    def dimension_complexity(self) -> str:
        return "O(n_samples * d) per generation"

    @property
    def space_complexity(self) -> str:
        return "O(n_samples * d)"

    @staticmethod
    def _rank_transform(rewards: np.ndarray) -> np.ndarray:
        """Replace rewards with centred ranks in [-0.5, 0.5]."""
        n = len(rewards)
        ranks = np.empty(n)
        ranks[np.argsort(rewards)] = np.arange(n)
        return ranks / (n - 1) - 0.5  # centred in [-0.5, 0.5]

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
        sigma = self.sigma_init * span

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
                history.extend([best_f, best_f])

            # rewards (negated: we minimise f, PEPG maximises reward)
            r_plus  = -f_plus
            r_minus = -f_minus

            if self.fitness_shaping:
                # rank-transform all 2N rewards together for a fair comparison
                all_r = np.concatenate([r_plus, r_minus])
                shaped = self._rank_transform(all_r)
                r_plus_s  = shaped[:self.n_samples]
                r_minus_s = shaped[self.n_samples:]
            else:
                b = (r_plus + r_minus).mean()
                r_plus_s  = r_plus  - b
                r_minus_s = r_minus - b

            mu_grad    = np.zeros(d)
            sigma_grad = np.zeros(d)
            for i in range(self.n_samples):
                advantage = (r_plus_s[i] - r_minus_s[i]) / 2.0
                mu_grad += advantage * epsilons[i]

                sym = (r_plus_s[i] + r_minus_s[i]) / 2.0
                sigma_grad += sym * (epsilons[i] ** 2 - 1.0)

            mu_grad    /= self.n_samples
            sigma_grad /= self.n_samples

            mu    = np.clip(mu + self.lr_mu * mu_grad, lower, upper)
            sigma = np.maximum(sigma + self.lr_sigma * sigma_grad * sigma, self.sigma_min)

        return RunResult(
            best_x=best_x,
            best_f=best_f,
            n_evaluations=evals,
            history_f=history,
        )
