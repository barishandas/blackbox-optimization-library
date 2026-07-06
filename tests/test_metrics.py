import numpy as np
import pytest
from bbolib.algorithms.base import RunResult
from bbolib.metrics import (
    success_rate, expected_running_time, median_feval_to_target,
    auc_convergence, convergence_curve,
)


def make_result(best_f, history=None, n_evals=None):
    if history is None:
        history = [best_f]
    return RunResult(
        best_x=np.zeros(2),
        best_f=best_f,
        n_evaluations=n_evals or len(history),
        history_f=history,
    )


TARGET = 1e-4


def test_success_rate_all_pass():
    results = [make_result(1e-8) for _ in range(10)]
    assert success_rate(results, TARGET) == 1.0


def test_success_rate_none_pass():
    results = [make_result(1.0) for _ in range(10)]
    assert success_rate(results, TARGET) == 0.0


def test_success_rate_partial():
    results = [make_result(1e-8)] * 3 + [make_result(1.0)] * 7
    assert success_rate(results, TARGET) == pytest.approx(0.3)


def test_ert_no_success():
    results = [make_result(1.0, n_evals=100) for _ in range(5)]
    assert expected_running_time(results, TARGET) == float("inf")


def test_ert_all_success():
    results = [make_result(1e-8, n_evals=200) for _ in range(5)]
    assert expected_running_time(results, TARGET) == pytest.approx(200.0)


def test_median_feval_found():
    hist = [10.0, 5.0, 1e-5, 1e-6]
    results = [make_result(1e-6, history=hist) for _ in range(5)]
    med = median_feval_to_target(results, TARGET)
    assert med == pytest.approx(3.0)  # index 2 (0-based) = 3rd eval


def test_auc_convergence_shape():
    history = list(np.linspace(1, 0, 100))
    results = [make_result(0.0, history=history) for _ in range(3)]
    auc = auc_convergence(results, budget=100)
    assert isinstance(auc, float)


def test_convergence_curve_shapes():
    history = list(np.linspace(1, 0.01, 200))
    results = [make_result(0.01, history=history) for _ in range(5)]
    x, med, pct = convergence_curve(results, budget=200)
    assert x.shape == (200,)
    assert med.shape == (200,)
    assert pct.shape == (2, 200)
