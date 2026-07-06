"""
Standard black-box benchmark functions.

Each function is defined on a canonical domain with a known global minimum.
All are minimization problems (f* = 0 after shifting unless noted).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class BenchmarkFunction:
    name: str
    func: callable
    bounds_1d: tuple[float, float]   # same bounds applied to every dimension
    f_opt: float                      # global minimum value
    x_opt_hint: str                   # description of where minimum lies
    group: str                        # separable / unimodal / multimodal / deceptive
    notes: str = ""

    def bounds(self, d: int) -> np.ndarray:
        lo, hi = self.bounds_1d
        return np.array([[lo, hi]] * d, dtype=float)

    def __call__(self, x: np.ndarray) -> float:
        return self.func(np.asarray(x, dtype=float))


# ---------------------------------------------------------------------------
# Separable
# ---------------------------------------------------------------------------

def _sphere(x):
    return float(np.sum(x ** 2))

def _step(x):
    return float(np.sum(np.floor(x + 0.5) ** 2))

def _sum_of_powers(x):
    d = len(x)
    return float(sum(abs(x[i]) ** (i + 2) for i in range(d)))

# ---------------------------------------------------------------------------
# Unimodal, ill-conditioned
# ---------------------------------------------------------------------------

def _rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

def _ellipsoid(x):
    d = len(x)
    c = np.logspace(0, 6, d)
    return float(np.sum(c * x ** 2))

def _discus(x):
    return float(1e6 * x[0] ** 2 + np.sum(x[1:] ** 2))

def _bent_cigar(x):
    return float(x[0] ** 2 + 1e6 * np.sum(x[1:] ** 2))

# ---------------------------------------------------------------------------
# Multimodal, structured
# ---------------------------------------------------------------------------

def _rastrigin(x):
    d = len(x)
    return float(10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))

def _ackley(x):
    d = len(x)
    a, b, c = 20.0, 0.2, 2 * np.pi
    return float(
        -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / d))
        - np.exp(np.sum(np.cos(c * x)) / d)
        + a + np.e
    )

def _griewank(x):
    d = len(x)
    return float(
        np.sum(x ** 2) / 4000
        - np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
        + 1
    )

def _schwefel(x):
    d = len(x)
    return float(418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

# ---------------------------------------------------------------------------
# Multimodal, weakly structured / deceptive
# ---------------------------------------------------------------------------

def _levy(x):
    w = 1 + (x - 1) / 4
    return float(
        np.sin(np.pi * w[0]) ** 2
        + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    )

def _styblinski_tang(x):
    return float(0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x) + 39.16599 * len(x))

def _drop_wave(x):
    # 2D only; generalised here as product of 1D drop waves
    norm = np.sqrt(np.sum(x ** 2))
    return float(-(1 + np.cos(12 * norm)) / (0.5 * norm ** 2 + 2))

def _michalewicz(x):
    m = 10
    i = np.arange(1, len(x) + 1)
    return float(-np.sum(np.sin(x) * np.sin(i * x ** 2 / np.pi) ** (2 * m)))

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FUNCTIONS: dict[str, BenchmarkFunction] = {
    # Separable
    "sphere": BenchmarkFunction(
        name="Sphere", func=_sphere, bounds_1d=(-5.12, 5.12),
        f_opt=0.0, x_opt_hint="x*=0", group="separable",
    ),
    "step": BenchmarkFunction(
        name="Step", func=_step, bounds_1d=(-5.12, 5.12),
        f_opt=0.0, x_opt_hint="|x_i| < 0.5", group="separable",
    ),
    "sum_of_powers": BenchmarkFunction(
        name="Sum of Powers", func=_sum_of_powers, bounds_1d=(-1, 1),
        f_opt=0.0, x_opt_hint="x*=0", group="separable",
    ),
    # Unimodal
    "rosenbrock": BenchmarkFunction(
        name="Rosenbrock", func=_rosenbrock, bounds_1d=(-5, 10),
        f_opt=0.0, x_opt_hint="x*=(1,...,1)", group="unimodal",
        notes="Narrow curved valley; ill-conditioned",
    ),
    "ellipsoid": BenchmarkFunction(
        name="Ellipsoid", func=_ellipsoid, bounds_1d=(-5, 5),
        f_opt=0.0, x_opt_hint="x*=0", group="unimodal",
        notes="Condition number 10^6",
    ),
    "discus": BenchmarkFunction(
        name="Discus", func=_discus, bounds_1d=(-5, 5),
        f_opt=0.0, x_opt_hint="x*=0", group="unimodal",
    ),
    "bent_cigar": BenchmarkFunction(
        name="Bent Cigar", func=_bent_cigar, bounds_1d=(-5, 5),
        f_opt=0.0, x_opt_hint="x*=0", group="unimodal",
        notes="Extreme condition number 10^6",
    ),
    # Multimodal structured
    "rastrigin": BenchmarkFunction(
        name="Rastrigin", func=_rastrigin, bounds_1d=(-5.12, 5.12),
        f_opt=0.0, x_opt_hint="x*=0", group="multimodal",
        notes="~10^d local minima on regular grid",
    ),
    "ackley": BenchmarkFunction(
        name="Ackley", func=_ackley, bounds_1d=(-32.768, 32.768),
        f_opt=0.0, x_opt_hint="x*=0", group="multimodal",
    ),
    "griewank": BenchmarkFunction(
        name="Griewank", func=_griewank, bounds_1d=(-600, 600),
        f_opt=0.0, x_opt_hint="x*=0", group="multimodal",
    ),
    "schwefel": BenchmarkFunction(
        name="Schwefel", func=_schwefel, bounds_1d=(-500, 500),
        f_opt=0.0, x_opt_hint="x*=420.9687...", group="multimodal",
        notes="Global minimum far from local optima — deceptive",
    ),
    # Deceptive / weakly structured
    "levy": BenchmarkFunction(
        name="Levy", func=_levy, bounds_1d=(-10, 10),
        f_opt=0.0, x_opt_hint="x*=(1,...,1)", group="deceptive",
    ),
    "styblinski_tang": BenchmarkFunction(
        name="Styblinski-Tang", func=_styblinski_tang, bounds_1d=(-5, 5),
        f_opt=0.0, x_opt_hint="x*=(-2.9035,...)", group="deceptive",
    ),
    "michalewicz": BenchmarkFunction(
        name="Michalewicz", func=_michalewicz, bounds_1d=(0, np.pi),
        f_opt=float("nan"),  # depends on d; ~-0.966*d
        x_opt_hint="near x_i=pi*i/(2m+2) for each i", group="deceptive",
        notes="Number of local minima grows with d",
    ),
}


def get_function(name: str) -> BenchmarkFunction:
    if name not in FUNCTIONS:
        raise KeyError(f"Unknown function '{name}'. Available: {list(FUNCTIONS)}")
    return FUNCTIONS[name]


def list_functions(group: str | None = None) -> list[str]:
    if group is None:
        return list(FUNCTIONS)
    return [k for k, v in FUNCTIONS.items() if v.group == group]
