import numpy as np
import math
from dataclasses import dataclass, field
from enum import Enum

from .constants import MU_0, EPS_0, ETA_0
from .functions import riccati_jn, riccati_h1, riccati_h2


class Mode(Enum):
    TM = "TM"
    TE = "TE"


@dataclass
class Layer:

    index: int
    r: float
    rel_permitivity: float

    @property
    def cols(self) -> range:
        return range(2*self.index, 2*self.index+4)

    @property
    def row_E(self) -> int:
        return 2*self.index + 1

    @property
    def row_H(self) -> int:
        return 2*self.index + 2

    def k(self, omega: float) -> float:
        return omega * np.sqrt(self.rel_permitivity * EPS_0 * MU_0)


@dataclass
class Onion:
    """Multilayer (onion) sphere with index convention::

          k_0      k_1    ... k_i        k_(N-1)      k_(N)
        ───·────)──────)─ ... ────)─ ... ────) · · · · ·  (Unbounded)
                r_0   r_1        r_i        r_(N-1)

        Layer i occupies r_{i-1} < r < r_i, with r_{-1} = 0.
        The outermost layer N is unbounded (r_N = ∞).
    """

    freq: float
    layers: list[Layer] = field(default_factory=list)



    @classmethod
    def from_arrays(cls, outer_radii: np.ndarray, permittivities: np.ndarray, frequency: float) -> "Onion":
        inner = [Layer(i, float(r), float(e))
                 for i, (r, e) in enumerate(zip(outer_radii, permittivities))]
        outer = Layer(len(inner), math.inf, 1.)
        return cls(freq=frequency, layers=inner + [outer])

    @property
    def angular_frequency(self) -> float:
        return 2 * np.pi * self.freq

    def r(self, i: int) -> float:
        return self.layers[i].r

    def mu(self, i: int) -> float:
        return MU_0  # Assumed Constant for this problem

    def k(self, i: int) -> float:

        return self.layers[i].k(self.angular_frequency)

    @property
    def outer_radii(self) -> np.ndarray:
        return np.array([layer.r for layer in self.layers])

    @property
    def permittivities(self) -> np.ndarray:
        return np.array([layer.rel_permitivity for layer in self.layers])

    def bc_E_TE_mode(self, i, m):
        ai = riccati_h1(m, self.k(i)*self.r(i))/self.mu(i)
        bi = riccati_h2(m, self.k(i)*self.r(i))/self.mu(i)
        aip1 = riccati_h1(m, self.k(i+1)*self.r(i))/self.mu(i+1)
        bip1 = riccati_h2(m, self.k(i+1)*self.r(i))/self.mu(i+1)
        return np.array([ai, bi, -aip1, -bip1])

    def bc_H_TE_mode(self, i, m):
        ai = riccati_h1(
            m, self.k(i)*self.r(i), deriv=True)/self.k(i)
        bi = riccati_h2(
            m, self.k(i)*self.r(i), deriv=True)/self.k(i)
        aip1 = riccati_h1(
            m, self.k(i+1)*self.r(i), deriv=True)/self.k(i+1)
        bip1 = riccati_h2(
            m, self.k(i+1)*self.r(i), deriv=True)/self.k(i+1)
        return np.array([ai, bi, -aip1, -bip1])

    def bc_E_TM_mode(self, i, m):
        ci = riccati_h1(m, self.k(i)*self.r(i))/self.k(i)
        di = riccati_h2(m, self.k(i)*self.r(i))/self.k(i)
        cip1 = riccati_h1(
            m, self.k(i+1)*self.r(i))/self.k(i+1)
        dip1 = riccati_h2(
            m, self.k(i+1)*self.r(i))/self.k(i+1)
        return np.array([ci, di, -cip1, -dip1])

    def bc_H_TM_mode(self, i, m):
        ci = riccati_h1(m, self.k(i)*self.r(i), deriv=True)/self.mu(i)
        di = riccati_h2(m, self.k(i)*self.r(i), deriv=True)/self.mu(i)
        cip1 = riccati_h1(
            m, self.k(i+1)*self.r(i), deriv=True)/self.mu(i+1)
        dip1 = riccati_h2(
            m, self.k(i+1)*self.r(i), deriv=True)/self.mu(i+1)
        return np.array([ci, di, -cip1, -dip1])

    def _rhs_prefactor(self, n):
        return 1j**(-n)*(2*n+1)/(n*(n+1))

    def rhs_E_TE(self, n):
        return self._rhs_prefactor(n)*riccati_jn(n, self.k(-1)*self.r(-2))/self.mu(-1)

    def rhs_H_TE(self, n):
        return self._rhs_prefactor(n)*riccati_jn(n, self.k(-1)*self.r(-2), derivative=True)/self.k(-1)

    def rhs_E_TM(self, n):
        return self._rhs_prefactor(n)*riccati_jn(n, self.k(-1)*self.r(-2))/self.k(-1)

    def rhs_H_TM(self, n):
        return self._rhs_prefactor(n)*riccati_jn(n, self.k(-1)*self.r(-2), derivative=True)/self.mu(-1)

    def solve(self, num_modes: int):
        num_layers = len(self.layers)
        a_TM = np.zeros((num_modes, num_layers), dtype=complex)
        b_TM = np.zeros((num_modes, num_layers), dtype=complex)
        a_TE = np.zeros((num_modes, num_layers), dtype=complex)
        b_TE = np.zeros((num_modes, num_layers), dtype=complex)

        for m in range(1,num_modes):
            A_TM, A_TE, rhs_TM, rhs_TE = self.assemble_one_mode(m)
            x_TM = np.linalg.solve(A_TM, rhs_TM)
            x_TE = np.linalg.solve(A_TE, rhs_TE)
            a_TM[m, :] = x_TM[0::2]
            b_TM[m, :] = x_TM[1::2]
            a_TE[m, :] = x_TE[0::2]
            b_TE[m, :] = x_TE[1::2]

        return a_TM,b_TM,a_TE,b_TE

    def assemble_one_mode(self, mode):
        """
        (a0_m,b0_m,a1_m,b1_m,...,al_m,bl_m)
        """
        num_layers: int = len(self.layers)
        num_unknowns: int = 2*num_layers
        A_TM = np.zeros((num_unknowns, num_unknowns), dtype=complex)
        A_TE = np.zeros((num_unknowns, num_unknowns), dtype=complex)
        rhs_TM = np.zeros((num_unknowns), dtype=complex)
        rhs_TE = np.zeros((num_unknowns), dtype=complex)

        # Enforce a^0 = b^0
        A_TM[0, :2] = [1, -1]
        A_TE[0, :2] = [1, -1]

        # Enforce interface conditions
        for layer in self.layers[:-1]:
            A_TM[layer.row_E, layer.cols] = self.bc_E_TM_mode(
                layer.index, mode)
            A_TM[layer.row_H, layer.cols] = self.bc_H_TM_mode(
                layer.index, mode)

            A_TE[layer.row_E, layer.cols] = self.bc_E_TE_mode(
                layer.index, mode)
            A_TE[layer.row_H, layer.cols] = self.bc_H_TE_mode(
                layer.index, mode)

        last_interface_layer: Layer = self.layers[-2]
        rhs_TM[last_interface_layer.row_E] = self.rhs_E_TM(mode)
        rhs_TM[last_interface_layer.row_H] = self.rhs_H_TM(mode)
        rhs_TE[last_interface_layer.row_E] = self.rhs_E_TE(mode)
        rhs_TE[last_interface_layer.row_H] = self.rhs_H_TE(mode)

        # Enforce a^l = 0
        A_TE[:, -2] = 0
        A_TM[:, -2] = 0
        A_TE[-1, -2] = 1
        A_TM[-1, -2] = 1

        return A_TM, A_TE, rhs_TM, rhs_TE

    def solve_and_plot(self, num_modes, component="magnitude",plot_log=False):
        """Plot the scattered field on the xz and yz planes.

        Parameters
        ----------
        component : {"magnitude", "phi", "theta", "r"}
            Which field component to display.  "magnitude" shows |E| and |H|;
            the others show the named spherical component.
        """
        import matplotlib.pyplot as plt
        from .field import scattered_field

        _valid = {"magnitude", "phi", "theta", "r"}
        if component not in _valid:
            raise ValueError(f"component must be one of {_valid}, got {component!r}")

        a_TM, b_TM, a_TE, b_TE = self.solve(num_modes)

        # Scattered (outgoing h2) coefficients in the outer unbounded region
        a_sc = b_TM[:, -1]
        b_sc = b_TE[:, -1]
        k_outer = self.k(-1)

        # Outer radius of the last physical shell
        r_outer = self.r(-2)

        wavelength = 2 * np.pi / k_outer
        lim = 3.0 * wavelength
        N = 300
        u = np.linspace(-lim, lim, N)
        U, V = np.meshgrid(u, u)

        def _extract(E_r, E_t, E_p, H_r, H_t, H_p):
            if component == "magnitude":
                E = np.sqrt(np.real(E_r)**2 + np.real(E_t)**2 + np.real(E_p)**2)
                H = np.sqrt(np.real(H_r)**2 + np.real(H_t)**2 + np.real(H_p)**2)
            elif component == "phi":
                E, H = np.real(E_p), np.real(H_p)
            elif component == "theta":
                E, H = np.real(E_t), np.real(H_t)
            else:  # "r"
                E, H = np.real(E_r), np.real(H_r)
            return E, H

        def _sph_on_grid(horiz, vert, phi_fixed=None,theta_fixed=None):
            """Evaluate scattered_field in spherical coords on a 2-D grid.

            horiz / vert are the two Cartesian axes spanned by the grid.
            phi_fixed overrides the azimuth (used for the xz plane where y=0).
            theta_fixed overides the elevation (used for xy plane where z=0)
            """
            r = np.sqrt(horiz**2 + vert**2)
            if theta_fixed is not None:
                theta = np.full_like(r,theta_fixed)
            else:
                theta = np.where(r > 0, np.arccos(np.clip(vert / r, -1, 1)), 0.0)
            
            if phi_fixed is not None:
                phi = np.full_like(r, phi_fixed)
            else:
                phi = np.arctan2(vert, horiz)  # xy: φ varies with y/x
            return scattered_field(a_sc, b_sc, k_outer, r, theta, phi)

        def _20log10(Field):
            return 20*np.log10(Field)
        

        # E-plane: xz (y = 0, phi = 0 for x > 0)
        E_r, E_t, E_p, H_r, H_t, H_p = _sph_on_grid(U, V, phi_fixed=0.0)
        E_xz, H_xz = _extract(E_r, E_t, E_p, H_r, H_t, H_p)
        H_xz = ETA_0 * H_xz
        inside = U**2 + V**2 < r_outer**2
        E_xz[inside] = np.nan
        H_xz[inside] = np.nan

        # H-plane: yz (x = 0, phi = π/2 for y > 0)
        E_r, E_t, E_p, H_r, H_t, H_p = _sph_on_grid(U, V, phi_fixed=np.pi / 2)
        E_yz, H_yz = _extract(E_r, E_t, E_p, H_r, H_t, H_p)
        H_yz = ETA_0 * H_yz
        E_yz[inside] = np.nan
        H_yz[inside] = np.nan

        # xy (z = 0, theta = π/2)
        E_r, E_t, E_p, H_r, H_t, H_p = _sph_on_grid(U, V, theta_fixed=np.pi / 2)
        E_xy, H_xy = _extract(E_r, E_t, E_p, H_r, H_t, H_p)
        H_xy = ETA_0 * H_xy
        E_xy[inside] = np.nan
        H_xy[inside] = np.nan

        comp_label = {
            "magnitude": r"\mathbf{{F}}^\mathrm{{sc}}",
            "phi":       r"F^\mathrm{{sc}}_\phi",
            "theta":     r"F^\mathrm{{sc}}_\theta",
            "r":         r"F^\mathrm{{sc}}_r",
        }[component]

        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        planes = [
            (E_xz, H_xz, "xz (E-plane)", r"$x / \lambda$", r"$z / \lambda$"),
            (E_yz, H_yz, "yz (H-plane)", r"$y / \lambda$", r"$z / \lambda$"),
            (E_xy, H_xy, "xy (V-plane)", r"$x / \lambda$", r"$y / \lambda$")
        ]
        for row, (E_data, H_data, plane_label, xlabel, ylabel) in enumerate(planes):
            for col, (data, field_sym, unit, prefix) in enumerate([
                (E_data, "E", "V/m", ""),
                (H_data, "H", "V/m", r"\eta_0 "),
            ]):
                label = prefix + comp_label.replace("F", field_sym)
                flabel = rf"$\mathrm{{Re}}\{{{label}\}}$ ({unit})"
                ax = axes[row, col]
                if plot_log:
                    im = ax.pcolormesh(U / wavelength, V / wavelength, _20log10(data), shading="auto", cmap="inferno", vmin=-1, vmax=1)
                else: 
                    im = ax.pcolormesh(U / wavelength, V / wavelength, data, shading="auto", cmap="inferno", vmin=-1, vmax=1)
                ax.add_patch(
                    plt.Circle((0, 0), r_outer / wavelength, fill=False, color="white", lw=1.5, ls="--")
                )
                ax.set_aspect("equal")
                ax.set_title(f"{flabel}  [{plane_label}]")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                plt.colorbar(im, ax=ax)

        fig.suptitle(
            f"Scattered field  (f = {self.freq/1e9:.2f} GHz,  component = {component})",
            y=1.01,
        )
        fig.tight_layout()
        return fig

    def solve_and_plot_rcs_nearfield(self, num_modes: int = 50, label: str = ""):
        """Plot bistatic RCS (col 1) and near-field |E| (col 2) for three cut planes.

        Layout
        ------
        Row 1 : XY plane  (θ = π/2)
        Row 2 : XZ plane  (E-plane, φ = 0)
        Row 3 : YZ plane  (H-plane, φ = π/2)

        Column 1 : Bistatic RCS  10·log10(σ3D / π r3²)  [dB]  vs θ ∈ [0°, 180°]
        Column 2 : Near-field |E|  20·log10(|E|)  [dB(V/m)]
        """
        import matplotlib.pyplot as plt
        from .field import scattered_field
        from .rcs import bistatic_rcs

        # ── 1.  Solve ────────────────────────────────────────────────────────────
        a_TM, b_TM, a_TE, b_TE = self.solve(num_modes)

        b_sc  = b_TM[:, -1]          # TM scattered coefficients
        d_sc  = b_TE[:, -1]          # TE scattered coefficients
        k     = self.k(-1)            # wavenumber in outer (unbounded) medium
        r3    = self.r(-2)            # outer radius of last physical shell
        lam   = 2 * np.pi / k
        norm  = np.pi * r3**2         # normalisation for RCS

        # ── 2.  RCS curves ───────────────────────────────────────────────────────
        theta     = np.linspace(0, np.pi, 361)
        theta_deg = np.degrees(theta)

        # XY cut  →  θ = π/2,  φ sweeps 0 → π  (same as azimuth scan)
        rcs_xy = np.array([bistatic_rcs(k, np.pi / 2, t, b_sc, d_sc) for t in theta])

        # XZ cut  →  φ = 0,    θ sweeps 0 → π  (E-plane)
        rcs_xz = np.array([bistatic_rcs(k, t, 0.0,         b_sc, d_sc) for t in theta])

        # YZ cut  →  φ = π/2,  θ sweeps 0 → π  (H-plane)
        rcs_yz = np.array([bistatic_rcs(k, t, np.pi / 2,   b_sc, d_sc) for t in theta])

        def _to_dB(rcs):
            return 10 * np.log10(np.maximum(rcs / norm, 1e-20))

        # ── 3.  Near-field grids ─────────────────────────────────────────────────
        lim = 3.0 * lam
        N   = 300
        u   = np.linspace(-lim, lim, N)
        U, V = np.meshgrid(u, u)

        def _E_magnitude(horiz, vert, phi_fixed=None, theta_fixed=None):
            """Return 20·log10(|E_sc|) on a 2-D Cartesian grid."""
            r_grid = np.sqrt(horiz**2 + vert**2)

            if theta_fixed is not None:
                theta_grid = np.full_like(r_grid, theta_fixed)
            else:
                theta_grid = np.where(
                    r_grid > 0,
                    np.arccos(np.clip(vert / r_grid, -1, 1)),
                    0.0,
                )

            if phi_fixed is not None:
                phi_grid = np.full_like(r_grid, phi_fixed)
            else:
                phi_grid = np.arctan2(vert, horiz)

            E_r, E_t, E_p, *_ = scattered_field(b_sc, d_sc, k, r_grid,
                                                theta_grid, phi_grid)
            E_mag = np.sqrt(np.abs(E_r)**2 + np.abs(E_t)**2 + np.abs(E_p)**2)

            # Mask interior of sphere
            inside      = horiz**2 + vert**2 < r3**2
            E_mag[inside] = np.nan

            return 20 * np.log10(np.maximum(E_mag, 1e-30))

        E_xy_dB = _E_magnitude(U, V, theta_fixed=np.pi / 2)   # XY: z = 0
        E_xz_dB = _E_magnitude(U, V, phi_fixed=0.0)            # XZ: y = 0, E-plane
        E_yz_dB = _E_magnitude(U, V, phi_fixed=np.pi / 2)      # YZ: x = 0, H-plane

        # ── 4.  Plot ─────────────────────────────────────────────────────────────
        _pt = 1. / 72.27
        _tw = 455.24411 * _pt
        fig, axes = plt.subplots(3, 2, figsize=(_tw, _tw * 1.5))

        rcs_ylabel = (
            r"$10\log_{10}\!\left(\sigma_\mathrm{3D}\,/\,\pi r_3^2\right)$  (dB)"
        )

        plane_cfg = [
            (rcs_xy, r"$\phi$ (degrees)", r"RCS -- XY  ($\theta$ = $90^\circ$)",          E_xy_dB, r"$x/\lambda$", r"$y/\lambda$", r"$|E^{sc}|$ -- XY  (z = 0)"),
            (rcs_xz, r"$\theta$ (degrees)", r"RCS -- XZ  (E-plane, $\varphi$ = $0^\circ$)",  E_xz_dB, r"$x/\lambda$", r"$z/\lambda$", r"$|E^{sc}|$ -- XZ  (y = 0)"),
            (rcs_yz, r"$\theta$ (degrees)", r"RCS -- YZ  (H-plane, $\varphi$ = $90^\circ$)", E_yz_dB, r"$y/\lambda$", r"$z/\lambda$", r"$|E^{sc}|$ -- YZ  (x = 0)"),
        ]

        for row, (rcs, x_rcs, title_rcs, E_dB, h_lbl, v_lbl, title_nf) in enumerate(plane_cfg):

            # ── Column 0 : RCS ───────────────────────────────────────────────────
            ax = axes[row, 0]
            ax.plot(theta_deg, _to_dB(rcs), color="steelblue")
            ax.axvline(90, color="gray", linestyle=":", linewidth=0.8)
            ax.set_xlabel(x_rcs)
            ax.set_ylabel(rcs_ylabel)
            ax.set_title(title_rcs)
            ax.set_xlim(0, 180)
            ax.set_xticks(np.arange(0, 181, 30))

            # ── Column 1 : Near-field |E| ────────────────────────────────────────
            ax = axes[row, 1]
            vmin = np.nanpercentile(E_dB, 2)
            vmax = np.nanpercentile(E_dB, 98)
            im = ax.pcolormesh(
                U / lam, V / lam, E_dB,
                shading="auto", cmap="inferno",
                vmin=vmin, vmax=vmax,
            )
            ax.add_patch(
                plt.Circle((0, 0), r3 / lam,
                            fill=False, color="white", lw=1.5, ls="--")
            )
            ax.set_aspect("equal")
            ax.set_xlabel(h_lbl)
            ax.set_ylabel(v_lbl)
            ax.set_title(title_nf)
            plt.colorbar(im, ax=ax, label=r"$20\log_{10}|E^{sc}|$  (dB V/m)")

        title = rf"Bistatic RCS \& Near-Field  --  {label}  " \
                rf"($f_0 = {self.freq/1e9:.2f}$ GHz,  $r_3 = {r3*1e3:.1f}$ mm)"
        fig.suptitle(title, fontweight="bold")
        fig.tight_layout()
        return fig