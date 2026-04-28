import numpy as np
from scipy.special import lpmv

from .functions import riccati_h2
from .constants import ETA_0


def _dP1_dtheta(n: int, theta: np.ndarray) -> np.ndarray:
    """d/dθ of P_n^1(cos θ) via recurrence: [n cos θ P_n^1 - (n+1) P_{n-1}^1] / sin θ."""
    if n == 0:
        return np.zeros_like(theta, dtype=complex)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    pn = lpmv(1, n, cos_t)
    pn_prev = lpmv(1, n - 1, cos_t)  # P_0^1 = 0, so n=1 term is correct
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(
            np.abs(sin_t) > 1e-14,
            (n * cos_t * pn - (n + 1) * pn_prev) / sin_t,
            0.0,
        )
    return result


def _P1_over_sin(n: int, theta: np.ndarray) -> np.ndarray:
    """P_n^1(cos θ) / sin θ, handled at the poles."""
    if n == 0:
        return np.zeros_like(theta, dtype=complex)
    sin_t = np.sin(theta)
    pn = lpmv(1, n, np.cos(theta))
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(np.abs(sin_t) > 1e-14, pn / sin_t, 0.0)
    return result


def scattered_field(
    a: np.ndarray,
    b: np.ndarray,
    k: float,
    r: float,
    theta: float,
    phi: float,
    E0: float = 1.0,
) -> tuple:
    """Compute the scattered E and H field components (eqs. 7.4.34–7.4.39).

    Parameters
    ----------
    a : array-like
        TM (a_n) Mie coefficients, indexed from n=0.
    b : array-like
        TE (b_n) Mie coefficients, indexed from n=0.
    k : float
        Wave number.
    r, theta, phi : float or array
        Spherical coordinates of the evaluation point.

    Returns
    -------
    (E_r, E_theta, E_phi, H_r, H_theta, H_phi) : tuple of complex arrays
    """
    a = np.asarray(a, dtype=complex)
    b = np.asarray(b, dtype=complex)
    N = max(len(a), len(b))

    r = np.asarray(r, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    x = k * r  # kr
    H0 = E0 / ETA_0

    E_r = np.zeros_like(x, dtype=complex)
    E_theta = np.zeros_like(x, dtype=complex)
    E_phi = np.zeros_like(x, dtype=complex)
    H_r = np.zeros_like(x, dtype=complex)
    H_theta = np.zeros_like(x, dtype=complex)
    H_phi = np.zeros_like(x, dtype=complex)

    for n in range(N):
        an = a[n] if n < len(a) else 0.0
        bn = b[n] if n < len(b) else 0.0

        h2 = riccati_h2(n, x)
        h2p = riccati_h2(n, x, deriv=True)  # d/d(kr) Ĥ_n^(2)(kr)

        Pn1 = lpmv(1, n, np.cos(theta))
        dPn1 = _dP1_dtheta(n, theta)
        Pn1_s = _P1_over_sin(n, theta)

        # 7.4.34 / 7.4.37: radial components
        radial_factor = n * (n + 1) * h2 / x**2
        E_r += an * radial_factor * Pn1
        H_r += bn * radial_factor * Pn1

        # 7.4.35 / 7.4.38: theta components
        transverse = (an * 1j * h2p * dPn1 + bn * h2 * Pn1_s) / x
        E_theta -= transverse
        H_theta -= (an * h2 * Pn1_s + bn * 1j * h2p * dPn1) / x

        # 7.4.36 / 7.4.39: phi components
        E_phi += (an * 1j * h2p * Pn1_s + bn * h2 * dPn1) / x
        H_phi -= (an * h2 * dPn1 + bn * 1j * h2p * Pn1_s) / x

    E_r *= E0 * np.cos(phi) / 1j
    E_theta *= E0 * np.cos(phi)
    E_phi *= E0 * np.sin(phi)
    H_r *= H0 * np.sin(phi) / 1j
    H_theta *= H0 * np.sin(phi)
    H_phi *= H0 * np.cos(phi)

    return E_r, E_theta, E_phi, H_r, H_theta, H_phi


def scattered_field_xyz(
    a_TM: np.ndarray,
    a_TE: np.ndarray,
    k: float,
    x: float,
    y: float,
    z: float,
    E0: float = 1.0,
) -> tuple:
    """Scattered field at Cartesian point (x, y, z), returned as (Ex, Ey, Ez, Hx, Hy, Hz)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    E_r, E_t, E_p, H_r, H_t, H_p = scattered_field(a_TM, a_TE, k, r, theta, phi, E0)

    sin_t, cos_t = np.sin(theta), np.cos(theta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)

    # Spherical-to-Cartesian rotation: [r̂, θ̂, φ̂] → [x̂, ŷ, ẑ]
    Ex = sin_t * cos_p * E_r + cos_t * cos_p * E_t - sin_p * E_p
    Ey = sin_t * sin_p * E_r + cos_t * sin_p * E_t + cos_p * E_p
    Ez = cos_t         * E_r - sin_t         * E_t

    Hx = sin_t * cos_p * H_r + cos_t * cos_p * H_t - sin_p * H_p
    Hy = sin_t * sin_p * H_r + cos_t * sin_p * H_t + cos_p * H_p
    Hz = cos_t         * H_r - sin_t         * H_t

    return Ex, Ey, Ez, Hx, Hy, Hz


def make_A_r(a: np.ndarray, k: float, omega: float, E0: float = 1.0):
    """Return lambda (r, θ, φ) → A_r^sc.

    A_r^sc = (E₀ cos φ / ω) Σₙ aₙ Ĥ_n^(2)(kr) Pₙ¹(cos θ)
    """
    a = np.asarray(a)

    def _A_r(r, theta, phi):
        x = k * np.asarray(r)
        theta = np.asarray(theta)
        total = sum(
            a[n] * riccati_h2(n, x) * lpmv(1, n, np.cos(theta))
            for n in range(len(a))
        )
        return E0 * np.cos(phi) / omega * total

    return _A_r


def make_F_r(b: np.ndarray, k: float, omega: float, E0: float = 1.0):
    """Return lambda (r, θ, φ) → F_r^sc.

    F_r^sc = (H₀ sin φ / ω) Σₙ bₙ Ĥ_n^(2)(kr) Pₙ¹(cos θ),  H₀ = E₀/η₀
    """
    b = np.asarray(b)
    H0 = E0 / ETA_0

    def _F_r(r, theta, phi):
        x = k * np.asarray(r)
        theta = np.asarray(theta)
        total = sum(
            b[n] * riccati_h2(n, x) * lpmv(1, n, np.cos(theta))
            for n in range(len(b))
        )
        return H0 * np.sin(phi) / omega * total

    return _F_r
