import numpy as np
import pytest
import dionpy
from dionpy.onion import Onion


def test_version():
    assert dionpy.__version__ == "0.1.0"


def test_assemble_a():
    r_a = np.array([3e-3, 4e-3, 12e-3])
    eps_a = np.array([50, 1, 4])
    Onion.from_arrays(r_a, eps_a, frequency=10e9).assemble(num_modes=3)


def test_assemble_b():
    r_b = np.array([20e-3, 27e-3, 81e-3])
    eps_b = np.array([50, 1, 4])
    Onion.from_arrays(r_b, eps_b, frequency=1.5e9).assemble(num_modes=3)
