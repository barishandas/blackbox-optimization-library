import numpy as np
import pytest
from bbolib.benchmarks import get_function, list_functions, FUNCTIONS


def test_all_functions_callable():
    for name in list_functions():
        fn = get_function(name)
        x = np.zeros(5)
        val = fn(x)
        assert isinstance(val, float)


def test_sphere_minimum():
    fn = get_function("sphere")
    assert fn(np.zeros(10)) == pytest.approx(0.0, abs=1e-10)


def test_rastrigin_minimum():
    fn = get_function("rastrigin")
    assert fn(np.zeros(10)) == pytest.approx(0.0, abs=1e-10)


def test_ackley_minimum():
    fn = get_function("ackley")
    assert fn(np.zeros(10)) == pytest.approx(0.0, abs=1e-8)


def test_rosenbrock_minimum():
    fn = get_function("rosenbrock")
    assert fn(np.ones(10)) == pytest.approx(0.0, abs=1e-10)


def test_bounds_shape():
    fn = get_function("sphere")
    for d in [2, 5, 10, 20]:
        b = fn.bounds(d)
        assert b.shape == (d, 2)
        assert np.all(b[:, 0] < b[:, 1])


def test_unknown_function_raises():
    with pytest.raises(KeyError):
        get_function("does_not_exist")


def test_list_functions_by_group():
    sep = list_functions(group="separable")
    assert "sphere" in sep
    multi = list_functions(group="multimodal")
    assert "rastrigin" in multi
    for name in sep:
        assert FUNCTIONS[name].group == "separable"
