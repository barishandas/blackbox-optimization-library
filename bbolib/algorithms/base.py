from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


@dataclass
class RunResult:
    """Outcome of a single algorithm run."""
    best_x: np.ndarray
    best_f: float
    n_evaluations: int
    history_f: list[float] = field(default_factory=list)  # best-so-far per evaluation
    converged: bool = False
    message: str = ""


class BBOAlgorithm(ABC):
    """Base class for all black-box optimizers in this library."""

    name: str = "unnamed"

    @abstractmethod
    def minimize(
        self,
        func: Callable[[np.ndarray], float],
        bounds: np.ndarray,
        budget: int,
        seed: int = 0,
    ) -> RunResult:
        """
        Minimize `func` within `bounds` using at most `budget` evaluations.

        Parameters
        ----------
        func    : objective function f(x) -> float (lower is better)
        bounds  : shape (d, 2) array of [lower, upper] per dimension
        budget  : maximum number of function evaluations
        seed    : random seed for reproducibility
        """

    @property
    def dimension_complexity(self) -> str:
        """Human-readable time complexity string, e.g. 'O(d^2 * lambda)'."""
        return "unknown"

    @property
    def space_complexity(self) -> str:
        return "unknown"
