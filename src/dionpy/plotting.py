import matplotlib.pyplot as plt
import numpy as np

from .constants import C_0, ETA_0
from .onion import Onion
from .field import scattered_field
from .rcs import bistatic_rcs

# ---------------------------------------------------------------------------
# Module-level plot helpers
# ---------------------------------------------------------------------------

_DEFAULT_COLORS = ["steelblue", "darkorange", "seagreen", "crimson", "purple"]

_pt = 1.0 / 72.27
_TW = 455.24411 * _pt          # text width in inches
_GOLDEN = (1 + 5**0.5) / 2


def plot_nearfield(onion: Onion, num_modes: int = 50, title: str = "") -> "Figure":
    """Near-field |E^sc| in dB for three cut planes (XZ, YZ, XY).

    Parameters
    ----------
    onion : Onion
        Solved geometry.
    num_modes : int
        Number of spherical modes.
    title : str
        Figure suptitle; auto-generated when empty.
    """

    a_TM, b_TM, a_TE, b_TE = onion.solve(num_modes)
    b_sc = b_TM[:, -1]
    d_sc = b_TE[:, -1]
    k = onion.k(-1)
    r3 = onion.r(-2)
    lam = 2 * np.pi / k

    lim = 3.0 * lam
    N = 300
    u = np.linspace(-lim, lim, N)
    U, V = np.meshgrid(u, u)
    inside = U**2 + V**2 < r3**2

    def _E_dB(horiz, vert, phi_fixed=None, theta_fixed=None):
        r_grid = np.sqrt(horiz**2 + vert**2)
        if theta_fixed is not None:
            th = np.full_like(r_grid, theta_fixed)
        else:
            th = np.where(r_grid > 0, np.arccos(
                np.clip(vert / r_grid, -1, 1)), 0.0)
        if phi_fixed is not None:
            ph = np.full_like(r_grid, phi_fixed)
        else:
            ph = np.arctan2(vert, horiz)
        E_r, E_t, E_p, *_ = scattered_field(b_sc, d_sc, k, r_grid, th, ph)
        E_mag = np.sqrt(np.abs(E_r)**2 + np.abs(E_t)**2 + np.abs(E_p)**2)
        E_mag[inside] = np.nan
        return 20 * np.log10(np.maximum(E_mag, 1e-30))

    panels = [
        (_E_dB(U, V, phi_fixed=0.0),         r"$x/\lambda$",
         r"$z/\lambda$", r"XZ  (E-plane)"),
        (_E_dB(U, V, phi_fixed=np.pi / 2),
         r"$y/\lambda$", r"$z/\lambda$", r"YZ  (H-plane)"),
        (_E_dB(U, V, theta_fixed=np.pi / 2),
         r"$x/\lambda$", r"$y/\lambda$", r"XY  (equatorial)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(_TW, _TW * 1.2))
    for ax, (E_dB, xlabel, ylabel, plane_title) in zip(axes, panels):
        vmin = np.nanpercentile(E_dB, 5)
        vmax = np.nanpercentile(E_dB, 95)
        im = ax.pcolormesh(U / lam, V / lam, E_dB, shading="auto", cmap="inferno",
                           vmin=vmin, vmax=vmax, rasterized=True)
        ax.add_patch(plt.Circle((0, 0), r3 / lam, fill=False,
                     color="white", lw=1.5, ls="--"))
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(plane_title)
        plt.colorbar(im, ax=ax, label=r"$20\log_{10}|E^{sc}|$  (dB V/m)")

    fig.suptitle(
        title or rf"Near-Field $|E^{{sc}}|$  ($f = {onion.freq/1e9:.2f}$ GHz)",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_bistatic_rcs(configs, num_modes: int = 30, norm: str = "r3",
                      ylim=None, colors=None, title: str = "") -> "Figure":
    """Bistatic RCS vs theta for E-plane and H-plane.

    Parameters
    ----------
    configs : list of dict
        Each dict must have keys ``'r'`` (radii array), ``'eps'`` (permittivities
        array), ``'freq'`` (Hz), and optionally ``'label'``.
    num_modes : int
        Number of spherical modes.
    norm : {'r3', 'lambda'}
        Normalise by ``pi * r3^2`` or by ``lambda^2``.
    ylim : tuple, optional
        ``(ymin, ymax)`` in dB for both axes.
    colors : list, optional
        Line colours; defaults to :data:`_DEFAULT_COLORS`.
    title : str
        Figure suptitle; auto-generated when empty.
    """

    if colors is None:
        colors = _DEFAULT_COLORS

    theta = np.linspace(0, np.pi, 361)
    theta_deg = np.degrees(theta)

    fig, axes = plt.subplots(2, 1, figsize=(_TW, _TW * _GOLDEN))

    ylabel = None
    for cfg, color in zip(configs, colors):
        r, eps, freq = np.asarray(cfg["r"]), np.asarray(
            cfg["eps"]), cfg["freq"]
        r3 = r[-1]
        sphere = Onion.from_arrays(r, eps, freq)
        _, b_TM, _, b_TE = sphere.solve(num_modes)
        b_sc = b_TM[:, -1]
        d_sc = b_TE[:, -1]
        k = sphere.k(-1)
        lam = 2 * np.pi / k

        if norm == "r3":
            normalization = np.pi * r3**2
            ylabel = r"$10\log_{10}\!\left(\sigma_\mathrm{3D}\,/\,\pi r_3^2\right)$  (dB)"
        else:
            normalization = lam**2
            ylabel = r"RCS / $\lambda^2$  (dB)"

        rcs_E = np.array([bistatic_rcs(k, t, 0.0,       b_sc, d_sc)
                         for t in theta])
        rcs_H = np.array([bistatic_rcs(k, t, np.pi / 2, b_sc, d_sc)
                         for t in theta])
        label = cfg.get("label", "")

        for ax, rcs in zip(axes, [rcs_E, rcs_H]):
            ax.plot(theta_deg, 10 * np.log10(np.maximum(rcs / normalization, 1e-20)),
                    color=color, label=label)

    for ax, plane in zip(axes, [r"E-Plane  ($\phi$ = $0^\circ$)",
                                r"H-Plane  ($\phi$ = $90^\circ$)"]):
        ax.set_xlabel(r"$\theta$  (degrees)")
        ax.set_ylabel(ylabel)
        ax.set_title(rf"Bistatic RCS -- {plane}")
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend()

    if title:
        fig.suptitle(title, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_monostatic_vs_r(permittivities, freq, r_fractions=None,
                         r_over_lam=None, num_modes: int = 50, title: str = "") -> "Figure":
    """Monostatic RCS normalised by pi*r_outer^2 vs r_outer/lambda.

    Parameters
    ----------
    permittivities : float or array-like
        Relative permittivities for each layer (outermost last).
    freq : float
        Frequency in Hz.
    r_fractions : array-like, optional
        Relative layer radii normalised so the outermost = 1.0.  Defaults
        to a single-layer sphere.
    r_over_lam : array-like, optional
        Sweep values for r_outer / lambda.  Defaults to 0.01 … 2.0.
    num_modes : int
        Number of spherical modes.
    title : str
        Axes title; auto-generated when empty.
    """

    permittivities = np.atleast_1d(np.asarray(permittivities, dtype=float))
    n_layers = len(permittivities)

    if r_fractions is None:
        r_fractions = np.ones(n_layers)
    r_fractions = np.asarray(r_fractions, dtype=float)
    r_fractions = r_fractions / r_fractions[-1]

    if r_over_lam is None:
        r_over_lam = np.linspace(0.01, 4.0, 400)
    r_over_lam = np.asarray(r_over_lam)

    lam = C_0 / freq
    rcs_mono = np.zeros(len(r_over_lam))

    for i, al in enumerate(r_over_lam):
        a_outer = al * lam
        radii = r_fractions * a_outer
        sphere = Onion.from_arrays(radii, permittivities, freq)
        _, b_TM, _, b_TE = sphere.solve(num_modes)
        b_sc = b_TM[:, -1]
        d_sc = b_TE[:, -1]
        k = sphere.k(-1)
        rcs_mono[i] = bistatic_rcs(k, np.pi, 0.0, b_sc, d_sc)

    a_phys = r_over_lam * lam
    rcs_dB = 10 * np.log10(np.maximum(rcs_mono / (np.pi * a_phys**2), 1e-20))

    fig, ax = plt.subplots(figsize=(_TW, _TW / _GOLDEN))
    ax.plot(r_over_lam, rcs_dB)
    ax.set_xlabel(r"$r/\lambda$")
    ax.set_ylabel(r"$\sigma_{3D} / \pi r^2$  (dB)")
    ax.set_xlim(r_over_lam[0], r_over_lam[-1])
    ax.set_xticks(np.arange(0, r_over_lam[-1] + 0.01, 0.5))

    eps_str = r",\,".join(str(int(e) if e == int(e) else e)
                          for e in permittivities)
    ax.set_title(title or rf"Monostatic RCS  ($\varepsilon_r = {eps_str}$)")
    fig.tight_layout()
    return fig



def plot_forward_bw(configs, bw_frac: float = 0.3, n_freqs: int = 400,
                    num_modes: int = 50, colors=None, title: str = "") -> "Figure":
    """Forward scattering (theta = 0) vs frequency over a fractional bandwidth.

    Parameters
    ----------
    configs : list of dict
        Each dict must have keys ``'r'``, ``'eps'``, ``'freq'``, and optionally
        ``'label'``.  One subplot is created per config.
    bw_frac : float
        Bandwidth as a fraction of the centre frequency.
    n_freqs : int
        Number of frequency samples.
    num_modes : int
        Number of spherical modes.
    colors : list, optional
        Line colours; defaults to :data:`_DEFAULT_COLORS`.
    title : str
        Figure suptitle; auto-generated when empty.
    """

    if colors is None:
        colors = _DEFAULT_COLORS

    n_cfg = len(configs)
    fig, axes = plt.subplots(n_cfg, 1, figsize=(_TW, _TW / _GOLDEN * n_cfg))
    if n_cfg == 1:
        axes = [axes]

    peak_freqs = []
    for ax, cfg, color in zip(axes, configs, colors):
        f0 = cfg["freq"]
        r = np.asarray(cfg["r"])
        eps = np.asarray(cfg["eps"])
        r3 = r[-1]
        BW = bw_frac * f0
        f_lo, f_hi = f0 - BW / 2, f0 + BW / 2
        freqs = np.linspace(f_lo, f_hi, n_freqs)
        x_axis = (freqs - f0) / BW
        rcs_fwd = np.zeros(n_freqs)

        for i, freq in enumerate(freqs):
            sphere = Onion.from_arrays(r, eps, freq)
            _, b_TM, _, b_TE = sphere.solve(num_modes)
            b_sc = b_TM[:, -1]
            d_sc = b_TE[:, -1]
            k = sphere.k(-1)
            rcs_fwd[i] = bistatic_rcs(k, 0.0, 0.0, b_sc, d_sc)

        norm = np.pi * r3**2
        rcs_fwd_dB = 10 * np.log10(np.maximum(rcs_fwd / norm, 1e-20))

        peak_idx = np.argmax(rcs_fwd_dB)
        peak_f_GHz = freqs[peak_idx] / 1e9
        peak_freqs.append(float(freqs[peak_idx]))

        ax.plot(x_axis, rcs_fwd_dB, color=color,
                label=r"$\sigma_{3D}(\theta=0^\circ)$")
        ax.plot(x_axis[peak_idx], rcs_fwd_dB[peak_idx], marker="*", markersize=8,
                color="crimson", zorder=5, label=rf"Peak: ${peak_f_GHz:.3f}$ GHz")
        ax.axvline(0.0, color="k", linestyle=":", linewidth=0.8,
                   label=rf"$f_0 = {f0/1e9:.2f}$ GHz")
        ax.axvspan(-0.5, 0.5, alpha=0.05, color=color)

        ax_top = ax.twiny()
        ax_top.set_xlim(f_lo / 1e9, f_hi / 1e9)
        ax_top.set_xlabel(r"$f$  (GHz)")
        tick_freqs = np.linspace(f_lo, f_hi, 5)
        ax_top.set_xticks(tick_freqs / 1e9)
        ax_top.set_xticklabels([f"{f/1e9:.3f}" for f in tick_freqs])

        layer_str = ",   ".join(
            rf"$\varepsilon_r={e}$, $r={ri*1e3:.0f}$ mm"
            for e, ri in zip(eps, r)
        )
        ax.set_xlabel(r"$(f - f_0)\;/\;\mathrm{BW}$")
        ax.set_ylabel(
            r"$10\log_{10}\!\left(\sigma_\mathrm{3D}\,/\,\pi r_3^2\right)$  (dB)")
        ax.set_title(f'{cfg.get("label", "")}:  {layer_str}')
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks(np.linspace(-0.5, 0.5, 11))
        ax.legend(loc="lower left")

    fig.suptitle(
        title or rf"Forward Scattering ($\theta=0^\circ$) -- BW = {bw_frac*100:.0f}\% of $f_0$",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig, peak_freqs


def plot_rcs_nearfield(onion: Onion, num_modes: int = 50, label: str = "") -> "Figure":
    """Bistatic RCS (col 1) and near-field |E| (col 2) for three cut planes.

    Layout: rows = XY, XZ, YZ planes; col 0 = RCS vs angle; col 1 = near-field.
    """
    _, b_TM, _, b_TE = onion.solve(num_modes)
    b_sc = b_TM[:, -1]
    d_sc = b_TE[:, -1]
    k = onion.k(-1)
    r3 = onion.r(-2)
    lam = 2 * np.pi / k
    norm = np.pi * r3**2

    theta = np.linspace(0, np.pi, 361)
    theta_deg = np.degrees(theta)

    rcs_xy = np.array([bistatic_rcs(k, np.pi / 2, t, b_sc, d_sc) for t in theta])
    rcs_xz = np.array([bistatic_rcs(k, t, 0.0, b_sc, d_sc) for t in theta])
    rcs_yz = np.array([bistatic_rcs(k, t, np.pi / 2, b_sc, d_sc) for t in theta])

    def _to_dB(rcs):
        return 10 * np.log10(np.maximum(rcs / norm, 1e-20))

    lim = 3.0 * lam
    N = 300
    u = np.linspace(-lim, lim, N)
    U, V = np.meshgrid(u, u)
    inside = U**2 + V**2 < r3**2

    def _E_dB(horiz, vert, phi_fixed=None, theta_fixed=None):
        r_grid = np.sqrt(horiz**2 + vert**2)
        if theta_fixed is not None:
            th = np.full_like(r_grid, theta_fixed)
        else:
            th = np.where(r_grid > 0, np.arccos(np.clip(vert / r_grid, -1, 1)), 0.0)
        if phi_fixed is not None:
            ph = np.full_like(r_grid, phi_fixed)
        else:
            ph = np.arctan2(vert, horiz)
        E_r, E_t, E_p, *_ = scattered_field(b_sc, d_sc, k, r_grid, th, ph)
        E_mag = np.sqrt(np.abs(E_r)**2 + np.abs(E_t)**2 + np.abs(E_p)**2)
        E_mag[inside] = np.nan
        return 20 * np.log10(np.maximum(E_mag, 1e-30))

    E_xy_dB = _E_dB(U, V, theta_fixed=np.pi / 2)
    E_xz_dB = _E_dB(U, V, phi_fixed=0.0)
    E_yz_dB = _E_dB(U, V, phi_fixed=np.pi / 2)

    fig, axes = plt.subplots(3, 2, figsize=(_TW, _TW * 1.5))
    rcs_ylabel = r"$10\log_{10}\!\left(\sigma_\mathrm{3D}\,/\,\pi r_3^2\right)$  (dB)"

    plane_cfg = [
        (rcs_xy, r"$\phi$ (degrees)", r"RCS -- XY  ($\theta$ = $90^\circ$)",
         E_xy_dB, r"$x/\lambda$", r"$y/\lambda$", r"$|E^{sc}|$ -- XY  (z = 0)"),
        (rcs_xz, r"$\theta$ (degrees)", r"RCS -- XZ  (E-plane, $\phi$ = $0^\circ$)",
         E_xz_dB, r"$x/\lambda$", r"$z/\lambda$", r"$|E^{sc}|$ -- XZ  (y = 0)"),
        (rcs_yz, r"$\theta$ (degrees)", r"RCS -- YZ  (H-plane, $\phi$ = $90^\circ$)",
         E_yz_dB, r"$y/\lambda$", r"$z/\lambda$", r"$|E^{sc}|$ -- YZ  (x = 0)"),
    ]

    for row, (rcs, x_rcs, title_rcs, E_dB, h_lbl, v_lbl, title_nf) in enumerate(plane_cfg):
        ax = axes[row, 0]
        ax.plot(theta_deg, _to_dB(rcs), color="steelblue")
        ax.axvline(90, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel(x_rcs)
        ax.set_ylabel(rcs_ylabel)
        ax.set_title(title_rcs)
        ax.set_xlim(0, 180)
        ax.set_xticks(np.arange(0, 181, 30))

        ax = axes[row, 1]
        vmin = np.nanpercentile(E_dB, 5)
        vmax = np.nanpercentile(E_dB, 95)
        im = ax.pcolormesh(U / lam, V / lam, E_dB, shading="auto", cmap="inferno",
                           vmin=vmin, vmax=vmax, rasterized=True)
        ax.add_patch(plt.Circle((0, 0), r3 / lam, fill=False, color="white", lw=1.5, ls="--"))
        ax.set_aspect("equal")
        ax.set_xlabel(h_lbl)
        ax.set_ylabel(v_lbl)
        ax.set_title(title_nf)
        plt.colorbar(im, ax=ax, label=r"$20\log_{10}|E^{sc}|$  (dB V/m)")

    fig.suptitle(
        rf"Bistatic RCS \& Near-Field  --  {label}  "
        rf"($f_0 = {onion.freq/1e9:.2f}$ GHz,  $r_3 = {r3*1e3:.1f}$ mm)",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_field_snapshots(onion: Onion, num_modes: int = 50, title: str = "") -> "Figure":
    """Six-panel near-field figure (Jin Fig. 7.13 style).

    3 rows × 2 columns:
      col 0 – E-plane (xz, φ=0),  component ηH_φ
      col 1 – H-plane (yz, φ=π/2), component E_φ

      row 0 – snapshot (Re) of scattered field
      row 1 – snapshot (Re) of total field
      row 2 – magnitude of total field

    All quantities normalised by the incident electric field amplitude E0 = 1.
    """
    a_TM, b_TM, a_TE, b_TE = onion.solve(num_modes)
    b_sc = b_TM[:, -1]
    d_sc = b_TE[:, -1]
    k = onion.k(-1)
    r3 = onion.r(-2)
    lam = 2 * np.pi / k

    lim = 3.0 * lam
    N = 300
    u = np.linspace(-lim, lim, N)
    Hg, Vg = np.meshgrid(u, u)          # horizontal / vertical spatial grid
    inside = Hg**2 + Vg**2 < r3**2
    r_grid = np.sqrt(Hg**2 + Vg**2)
    th = np.where(r_grid > 0, np.arccos(np.clip(Vg / r_grid, -1, 1)), 0.0)
    inc_phase = np.exp(1j * k * Vg)     # e^{ikz}, z = vertical axis

    # Scattered field in each plane
    _, _, _, _, _, H_phi_sc = scattered_field(b_sc, d_sc, k, r_grid, th,
                                              np.zeros_like(th))
    _, _, E_phi_sc, _, _, _ = scattered_field(b_sc, d_sc, k, r_grid, th,
                                              np.full_like(th, np.pi / 2))

    # η₀-normalised fields and totals
    # At φ=0:   η₀ H_φ^inc = E0 e^{ikz}
    # At φ=π/2: E_φ^inc    = -E0 e^{ikz}
    eta_Hphi_sc  = ETA_0 * H_phi_sc
    eta_Hphi_tot = eta_Hphi_sc + inc_phase
    E_phi_tot    = E_phi_sc - inc_phase

    def _re(arr):
        return np.where(inside, np.nan, np.real(arr))

    def _abs(arr):
        return np.where(inside, np.nan, np.abs(arr))

    panels = [
        (_re(eta_Hphi_sc),   r"$\eta H_\phi^s$  (E-plane)",   r"$x/\lambda$", r"$z/\lambda$", -1, 1),
        (_re(E_phi_sc),      r"$E_\phi^s$  (H-plane)",         r"$y/\lambda$", r"$z/\lambda$", -1, 1),
        (_re(eta_Hphi_tot),  r"$\eta H_\phi$  (E-plane)",      r"$x/\lambda$", r"$z/\lambda$", -2, 2),
        (_re(E_phi_tot),     r"$E_\phi$  (H-plane)",            r"$y/\lambda$", r"$z/\lambda$", -2, 2),
        (_abs(eta_Hphi_tot), r"$|\eta H_\phi|$  (E-plane)",    r"$x/\lambda$", r"$z/\lambda$",  0, 2),
        (_abs(E_phi_tot),    r"$|E_\phi|$  (H-plane)",          r"$y/\lambda$", r"$z/\lambda$",  0, 2),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(_TW * 1.2, _TW * 1.6))
    for ax, (data, panel_title, xlabel, ylabel, vmin, vmax) in zip(axes.flat, panels):
        im = ax.pcolormesh(Hg / lam, Vg / lam, data, shading="auto", cmap="jet",
                           vmin=vmin, vmax=vmax, rasterized=True)
        ax.add_patch(plt.Circle((0, 0), r3 / lam, fill=False, color="white", lw=1.5, ls="--"))
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        plt.colorbar(im, ax=ax)

    fig.suptitle(
        title or rf"Near-Field Snapshots  ($f = {onion.freq/1e9:.2f}$ GHz,  $r_3 = {r3*1e3:.1f}$ mm)",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig
