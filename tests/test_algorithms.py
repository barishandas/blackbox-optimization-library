import numpy as np
import pytest
from bbolib.algorithms import CMAESAlgorithm, DEAlgorithm, PSOAlgorithm, RandomSearch
from bbolib.benchmarks import get_function

SPHERE = get_function("sphere")
BOUNDS_D5 = SPHERE.bounds(5)
BUDGET = 2000


@pytest.fixture(params=[
    CMAESAlgorithm(),
    DEAlgorithm(),
    PSOAlgorithm(n_particles=20),
    RandomSearch(),
])
def algo(request):
    return request.param


def test_returns_run_result(algo):
    result = algo.minimize(SPHERE, BOUNDS_D5, budget=BUDGET, seed=0)
    assert result.best_f is not None
    assert result.n_evaluations > 0
    assert result.n_evaluations <= BUDGET + 50  # small overrun allowed
    assert len(result.history_f) > 0


def test_history_is_non_increasing(algo):
    result = algo.minimize(SPHERE, BOUNDS_D5, budget=BUDGET, seed=1)
    hist = result.history_f
    for i in range(1, len(hist)):
        assert hist[i] <= hist[i - 1] + 1e-12, f"History not monotone at step {i}"


def test_best_x_within_bounds(algo):
    result = algo.minimize(SPHERE, BOUNDS_D5, budget=BUDGET, seed=2)
    lo, hi = BOUNDS_D5[:, 0], BOUNDS_D5[:, 1]
    assert np.all(result.best_x >= lo - 1e-8)
    assert np.all(result.best_x <= hi + 1e-8)


def test_sphere_converges():
    algo = CMAESAlgorithm()
    result = algo.minimize(SPHERE, BOUNDS_D5, budget=5000, seed=42)
    assert result.best_f < 1e-6, f"CMA-ES did not converge on Sphere (got {result.best_f})"


def test_deterministic_with_same_seed():
    algo = PSOAlgorithm(n_particles=20)
    r1 = algo.minimize(SPHERE, BOUNDS_D5, budget=500, seed=7)
    r2 = algo.minimize(SPHERE, BOUNDS_D5, budget=500, seed=7)
    assert r1.best_f == r2.best_f
