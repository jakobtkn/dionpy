from scipy.special import assoc_legendre_p_all
import numpy as np


def bistatic_rcs(k: float, theta: float, phi: float, b: list, d: list) -> float:
    """
    Compute the bistatic radar cross section (RCS).

    Parameters
    ----------
    k     : Wave number (float)
    theta : Polar angle in radians (float)
    phi   : Azimuthal angle in radians (float)
    b     : Expansion coefficients b_m, m = 1..N (list of complex)
    d     : Expansion coefficients d_m, m = 1..N (list of complex)

    Returns
    -------
    rcs : Bistatic RCS (float)
    """
    if len(b) != len(d):
        raise ValueError("b and d must have the same length.")

    N = len(b)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    sum1 = 0.0 + 0.0j
    sum2 = 0.0 + 0.0j

    Pm1_all = assoc_legendre_p_all(N-1, 1, cos_t, diff_n=1)

    if abs(sin_t) < 1e-9:
        dP_dtheta = -cos_t * Pm1_all[1,:,0]
    else:
        dP_dtheta = -sin_t * Pm1_all[1,:,1]
    Pm1_over_sinT = -Pm1_all[1,:,0]
    jm = 1j ** np.arange(0, N)
    sum1 = np.sum(jm * (b * dP_dtheta + d * Pm1_over_sinT))
    sum2 = np.sum(jm * (b * Pm1_over_sinT + d * dP_dtheta))

    rcs = (4.0 * np.pi / k**2) * (
        np.cos(phi)**2 * abs(sum1)**2 +
        np.sin(phi)**2 * abs(sum2)**2
    )
    return float(rcs)
