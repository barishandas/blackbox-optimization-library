import numpy as np
from .base import BBOAlgorithm, RunResult
from typing import Callable


class DEAlgorithm(BBOAlgorithm):
    """
    Differential Evolution with optional jDE self-adaptive parameter control
    (Brest et al. 2006) — adapts F and CR per individual rather than using
    fixed values, significantly improving performance on multimodal functions.

    Parameters
    ----------
    strategy    : DE variant (default 'rand1bin')
    mutation    : initial F scale factor (float or (min, max) dithering range)
    recombination : initial CR crossover rate
    popsize     : population size multiplier (actual NP = popsize * d)
    adaptive    : if True, use jDE self-adaptive F and CR per individual
    p_adapt_f   : probability of re-drawing F each generation (jDE)
    p_adapt_cr  : probability of re-drawing CR each generation (jDE)
    """

    name = "DE"

    def __init__(
        self,
        strategy: str = "currenttobest1bin",
        mutation: float | tuple[float, float] = 0.8,
        recombination: float = 0.9,
        popsize: int = 5,
        adaptive: bool = True,
        p_adapt_f: float = 0.1,
        p_adapt_cr: float = 0.1,
    ):
        self.strategy = strategy
        self.mutation = mutation
        self.recombination = recombination
        self.popsize = popsize
        self.adaptive = adaptive
        self.p_adapt_f = p_adapt_f
        self.p_adapt_cr = p_adapt_cr

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
        rng = np.random.default_rng(seed)
        bounds = np.asarray(bounds, dtype=float)
        d = bounds.shape[0]
        lower, upper = bounds[:, 0], bounds[:, 1]
        NP = max(4, self.popsize * d)

        # initialise population
        pop = rng.uniform(lower, upper, size=(NP, d))
        fitness = np.array([func(pop[i]) for i in range(NP)], dtype=float)
        evals = NP

        best_idx = int(np.argmin(fitness))
        best_f = float(fitness[best_idx])
        best_x = pop[best_idx].copy()
        history = list(np.minimum.accumulate([best_f] * NP))

        # per-individual F and CR for jDE
        F_arr  = np.full(NP, float(self.mutation) if not isinstance(self.mutation, tuple) else 0.5)
        CR_arr = np.full(NP, self.recombination)

        while evals + NP <= budget:
            new_pop = pop.copy()
            new_fit = fitness.copy()

            for i in range(NP):
                # jDE: randomly refresh F and CR
                if self.adaptive:
                    F_i  = rng.uniform(0.1, 0.9) if rng.random() < self.p_adapt_f  else F_arr[i]
                    CR_i = rng.uniform(0.0, 1.0) if rng.random() < self.p_adapt_cr else CR_arr[i]
                else:
                    F_i  = float(self.mutation) if not isinstance(self.mutation, tuple) else rng.uniform(*self.mutation)
                    CR_i = self.recombination

                # select mutation indices (all different from i)
                idxs = [j for j in range(NP) if j != i]
                a, b, c = rng.choice(idxs, size=3, replace=False)

                if self.strategy == "rand1bin":
                    mutant = pop[a] + F_i * (pop[b] - pop[c])
                elif self.strategy == "best1bin":
                    mutant = best_x + F_i * (pop[a] - pop[b])
                elif self.strategy == "rand2bin":
                    e = rng.choice([j for j in idxs if j not in (a, b, c)])
                    mutant = pop[a] + F_i * (pop[b] - pop[c]) + F_i * (pop[e] - pop[a])
                elif self.strategy == "currenttobest1bin":
                    mutant = pop[i] + F_i * (best_x - pop[i]) + F_i * (pop[a] - pop[b])
                else:
                    mutant = pop[a] + F_i * (pop[b] - pop[c])

                mutant = np.clip(mutant, lower, upper)

                # binomial crossover
                cross_mask = rng.random(d) < CR_i
                cross_mask[rng.integers(d)] = True  # guarantee at least one
                trial = np.where(cross_mask, mutant, pop[i])

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    new_pop[i] = trial
                    new_fit[i] = f_trial
                    if self.adaptive:
                        F_arr[i]  = F_i
                        CR_arr[i] = CR_i
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        best_idx = i

                history.append(best_f)

            pop = new_pop
            fitness = new_fit

        return RunResult(
            best_x=best_x,
            best_f=best_f,
            n_evaluations=evals,
            history_f=history,
        )
