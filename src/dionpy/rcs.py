from scipy.special import assoc_legendre_p
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
    near_pole = abs(sin_t) < 1e-9

    sum1 = 0.0 + 0.0j
    sum2 = 0.0 + 0.0j

    for idx in range(N):
        m = idx + 1

        res   = assoc_legendre_p(m, 1, cos_t, diff_n=1)
        Pm1   = res[0]
        dP_dz = res[1]

        if near_pole:
            sign = (-1) ** (m + 1) if cos_t < 0 else 1.0
            limit         = m * (m + 1) / 2.0
            Pm1_over_sinT = sign * limit
            dP_dtheta     = -sign * limit   # τ_n has opposite sign at θ=π
        else:
            dP_dtheta     = -sin_t * dP_dz
            Pm1_over_sinT = Pm1 / sin_t

        jm = 1j ** m
        sum1 += jm * (b[idx] * dP_dtheta     + d[idx] * Pm1_over_sinT)
        sum2 += jm * (b[idx] * Pm1_over_sinT + d[idx] * dP_dtheta)

    rcs = (4.0 * np.pi / k**2) * (
        np.cos(phi)**2 * abs(sum1)**2 +
        np.sin(phi)**2 * abs(sum2)**2
    )
    return float(rcs)