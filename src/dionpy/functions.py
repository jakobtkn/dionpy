from scipy.special import spherical_yn, spherical_jn
import numpy as np


def riccati_jn(n: int, x: np.ndarray, derivative: bool = False) -> np.ndarray:
    """Riccati-Bessel Ĵ_n(x) = x j_n(x), or its x-derivative."""
    if derivative:
        return spherical_jn(n, x) + x * spherical_jn(n, x, derivative=True)
    return x * spherical_jn(n, x)


def riccati_yn(n: int, x: np.ndarray, deriv: bool = False) -> np.ndarray:
    """Riccati-Neumann Ŷ_n(x) = -x y_n(x), or its x-derivative."""
    if deriv:
        return -(spherical_yn(n, x) + x * spherical_yn(n, x, derivative=True))
    return -x * spherical_yn(n, x)


def riccati_h1(n: int, x: np.ndarray, deriv: bool = False) -> np.ndarray:
    """Riccati-Hankel Ĥ_n^(1)(x) = x h_n^(1)(x)."""
    return riccati_jn(n, x, deriv) - 1j * riccati_yn(n, x, deriv)


def riccati_h2(n: int, x: np.ndarray, deriv: bool = False) -> np.ndarray:
    """Riccati-Hankel Ĥ_n^(2)(x) = x h_n^(2)(x)."""
    return riccati_jn(n, x, deriv) + 1j * riccati_yn(n, x, deriv)
