# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "dionpy",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from dionpy.onion import Onion, Layer
    from dionpy.constants import C_0, EPS_0
    import numpy as np
    import matplotlib.pyplot as plt
    from dionpy.rcs import bistatic_rcs
    from dionpy.field import scattered_field, scattered_field_xyz

    _pt = 1. / 72.27
    tw = 455.24411 * _pt
    golden = (1 + 5**0.5) / 2

    return C_0, Onion, bistatic_rcs, golden, mo, np, plt, tw


@app.cell
def _(plt):
    plt.style.use("../Plots/paper.mplstyle")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Examples
    """)
    return


@app.cell
def _(C_0, Onion, np):
    # From Project Description
    _r_a = np.array([3e-3, 4e-3, 12e-3])
    _eps_a = np.array([50, 1, 4])
    test_a = Onion.from_arrays(_r_a, _eps_a, 10e9)

    _r_b = np.array([20e-3, 27e-3, 81e-3])
    _eps_b = np.array([50, 1, 4])
    _test_b = Onion.from_arrays(_r_b, _eps_b, 1.5e9)

    # From J.-M. Jin, "THEORY AND COMPUTATION OF ELECTROMAGNETIC FIELDS".
    # Recreates Figure 7.13, 7.14 and 7.15
    _r_jin = np.array([C_0 / 400e9])
    _eps_jin = np.array([2.56])
    _test_jin = Onion.from_arrays(_r_jin, _eps_jin, 400e9)
    return (test_a,)


@app.cell
def _(plt, test_a, tw, golden):
    _A, _, _b, _ = test_a.assemble_one_mode(5)
    _fig, _axes = plt.subplots(1, 2, figsize=(tw, tw / golden))
    _axes[0].matshow(abs(_A) > 0)
    _axes[0].set_title("A")
    _axes[1].matshow(abs(_b[:, None]) > 0)
    _axes[1].set_title("b")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Example 7_14 and 7_15 from Cheng
    """)
    return


@app.cell
def _(C_0, Onion, bistatic_rcs, np, plt, tw, golden):
    _freq          = 400e9
    _radii_in_lambda = [1.0, 0.5, 0.2]
    _colors        = ['steelblue', 'darkorange', 'seagreen']
    _theta         = np.linspace(0, np.pi, 361)
    _theta_deg     = np.degrees(_theta)

    _fig, _axes = plt.subplots(1, 2, figsize=(tw, tw / golden))

    for _r_lambda, _color in zip(_radii_in_lambda, _colors):
        _lam0   = C_0 / _freq
        _radius = _r_lambda * _lam0

        _sphere = Onion.from_arrays(np.array([_radius]), np.array([2.56]), _freq)
        _, _b_TM, _, _b_TE = _sphere.solve(num_modes=30)

        _b_sc = _b_TM[:, -1]
        _d_sc = _b_TE[:, -1]
        _k    = _sphere.k(-1)
        _lam  = 2 * np.pi / _k

        _label = rf'r = {_r_lambda}$\lambda$'

        _rcs_E    = np.array([bistatic_rcs(_k, t, 0,        _b_sc, _d_sc) for t in _theta])
        _rcs_E_dB = 10 * np.log10(np.maximum(_rcs_E / _lam**2, 1e-20))
        _axes[0].plot(_theta_deg, _rcs_E_dB, color=_color, label=_label)

        _rcs_H    = np.array([bistatic_rcs(_k, t, np.pi/2, _b_sc, _d_sc) for t in _theta])
        _rcs_H_dB = 10 * np.log10(np.maximum(_rcs_H / _lam**2, 1e-20))
        _axes[1].plot(_theta_deg, _rcs_H_dB, color=_color, label=_label)

    for _ax, _plane in zip(_axes, [r'E-Plane  ($\varphi$ = $0^\circ$)', r'H-Plane  ($\varphi$ = $90^\circ$)']):
        _ax.set_xlabel(r'$\theta$ (degrees)')
        _ax.set_ylabel(r'RCS/$\lambda^2$ (dB)')
        _ax.set_title(rf'Bistatic RCS vs $\theta$ -- {_plane}')
        _ax.set_xlim(0, 180)
        _ax.set_ylim(-50, 25)
        _ax.set_xticks(np.arange(0, 181, 30))
        _ax.legend()

    _fig.suptitle(r'Bistatic Radar Cross Section  ($\varepsilon_r$ = 2.56,  f = 400 GHz)',
                  fontweight='bold', y=1.01)
    plt.tight_layout()
    _fig.savefig(r"../Plots/RCS_example_fig_7_14.pdf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Monostatic RCS vs $r/\lambda$ (Figure 7.15 reproduction)
    """)
    return


@app.cell
def _(C_0, Onion, bistatic_rcs, np, plt, tw, golden):
    _freq  = 400e9
    _lam0  = C_0 / _freq
    _eps_r = 2.56

    _r_over_lam = np.linspace(0.01, 2.0, 400)
    _rcs_mono   = np.zeros(len(_r_over_lam))

    for _i, _al in enumerate(_r_over_lam):
        _a      = _al * _lam0
        _sphere = Onion.from_arrays(np.array([_a]), np.array([_eps_r]), _freq)
        _, _b_TM, _, _b_TE = _sphere.solve(num_modes=50)
        _b_sc = _b_TM[:, -1]
        _d_sc = _b_TE[:, -1]
        _k    = _sphere.k(-1)
        _rcs_mono[_i] = bistatic_rcs(_k, np.pi, 0.0, _b_sc, _d_sc)

    _a_phys      = _r_over_lam * _lam0
    _rcs_norm    = _rcs_mono / (np.pi * _a_phys**2)
    _rcs_norm_dB = 10 * np.log10(np.maximum(_rcs_norm, 1e-20))

    _fig, _ax = plt.subplots(figsize=(tw / 2, tw / 2 / golden))
    _ax.plot(_r_over_lam, _rcs_norm_dB, color='k')
    _ax.set_xlabel(r'$r/\lambda$')
    _ax.set_ylabel(r'$\sigma_{3D} / \pi r^2$  (dB)')
    _ax.set_title(r'Monostatic RCS --- Dielectric Sphere  ($\varepsilon_r = 2.56$)')
    _ax.set_xlim(0, 2.0)
    _ax.set_ylim(-50, 20)
    _ax.set_xticks(np.arange(0, 2.1, 0.5))
    _ax.set_yticks(np.arange(-50, 21, 10))
    _ax.annotate('Rayleigh scattering region', xy=(0.02, -30),
                 xytext=(0.25, -30),
                 arrowprops=dict(arrowstyle='<-', color='k'))
    plt.tight_layout()
    _fig.savefig(r"../Plots/RCS_example_fig_7_15.pdf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Monostatic RCS of PEC (Figure 11-29 Balanis)
    """)
    return


@app.cell
def _(C_0, Onion, bistatic_rcs, np, plt, tw, golden):
    _freq  = 400e9
    _lam0  = C_0 / _freq
    _eps_r = 100000

    _r_over_lam = np.linspace(0.01, 2.0, 400)
    _rcs_mono   = np.zeros(len(_r_over_lam))

    for _i, _al in enumerate(_r_over_lam):
        _a      = _al * _lam0
        _sphere = Onion.from_arrays(np.array([_a]), np.array([_eps_r]), _freq)
        _, _b_TM, _, _b_TE = _sphere.solve(num_modes=50)
        _b_sc = _b_TM[:, -1]
        _d_sc = _b_TE[:, -1]
        _k    = _sphere.k(-1)
        _rcs_mono[_i] = bistatic_rcs(_k, np.pi, 0.0, _b_sc, _d_sc)

    _a_phys      = _r_over_lam * _lam0
    _rcs_norm    = _rcs_mono / (np.pi * _a_phys**2)
    _rcs_norm_dB = 10 * np.log10(np.maximum(_rcs_norm, 1e-20))

    _fig, _ax = plt.subplots(figsize=(tw / 2, tw / 2 / golden))
    _ax.plot(_r_over_lam, _rcs_norm_dB, color='k')
    _ax.set_xlabel(r'$r/\lambda$')
    _ax.set_ylabel(r'$\sigma_{3D} / \pi r^2$  (dB)')
    _ax.set_title(r'Monostatic RCS --- PEC Sphere')
    _ax.set_xlim(0, 2.0)
    _ax.set_ylim(-50, 20)
    _ax.set_xticks(np.arange(0, 2.1, 0.5))
    _ax.set_yticks(np.arange(-50, 21, 10))
    _ax.annotate('Rayleigh scattering region', xy=(0.02, -30),
                 xytext=(0.25, -30),
                 arrowprops=dict(arrowstyle='<-', color='k'))
    plt.tight_layout()
    _fig.savefig(r"../Plots/mono_RCS_PEC.pdf")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Exercise from Project 2 description
    1. Plot of Bistatic RCS as a function of theta in the XZ and YZ plane
    2. Plot of Forwards RCS as a function of frequency (XZ and YZ plane)
    3. Plot of 3D RCS ($10\log_{10}\sigma_{3D}/(\pi r^2)$) in XZ, YZ and XY plane as well as near field $20\log_{10}|E|$
    """)
    return


@app.cell
def _(Onion, bistatic_rcs, np, plt, tw, golden):
    _configs = [
        dict(label='Config A', r=np.array([3e-3,  4e-3, 12e-3]),
             eps=np.array([50, 1, 4]), f0=10e9),
        dict(label='Config B', r=np.array([20e-3, 27e-3, 81e-3]),
             eps=np.array([50, 1, 4]), f0=1.5e9),
    ]

    _colors    = ['steelblue', 'darkorange']
    _theta     = np.linspace(0, np.pi, 361)
    _theta_deg = np.degrees(_theta)

    _fig, _axes = plt.subplots(1, 2, figsize=(tw, tw / golden))

    for _cfg, _color in zip(_configs, _colors):
        _r   = _cfg['r']
        _eps = _cfg['eps']
        _f0  = _cfg['f0']
        _r3  = _r[-1]

        _sphere = Onion.from_arrays(_r, _eps, _f0)
        _, _b_TM, _, _b_TE = _sphere.solve(num_modes=30)

        _b_sc = _b_TM[:, -1]
        _d_sc = _b_TE[:, -1]
        _k    = _sphere.k(-1)
        _norm = np.pi * _r3**2

        _rcs_E    = np.array([bistatic_rcs(_k, t, 0,       _b_sc, _d_sc) for t in _theta])
        _rcs_E_dB = 10 * np.log10(np.maximum(_rcs_E / _norm, 1e-20))
        _axes[0].plot(_theta_deg, _rcs_E_dB, color=_color, label=_cfg['label'])

        _rcs_H    = np.array([bistatic_rcs(_k, t, np.pi/2, _b_sc, _d_sc) for t in _theta])
        _rcs_H_dB = 10 * np.log10(np.maximum(_rcs_H / _norm, 1e-20))
        _axes[1].plot(_theta_deg, _rcs_H_dB, color=_color, label=_cfg['label'])

    for _ax, _plane in zip(_axes, [r'E-Plane  ($\varphi$ = $0^\circ$)', r'H-Plane  ($\varphi$ = $90^\circ$)']):
        _ax.set_xlabel(r'$\theta$  (degrees)')
        _ax.set_ylabel(r'$10\log_{10}\!\left(\sigma_\mathrm{3D}\,/\,\pi r_3^2\right)$  (dB)')
        _ax.set_title(rf'Bistatic RCS -- {_plane}')
        _ax.set_xlim(0, 180)
        _ax.set_xticks(np.arange(0, 181, 30))
        _ax.legend()

    _fig.suptitle(r'Bistatic RCS -- Config A vs Config B', fontweight='bold')
    plt.tight_layout()
    _fig.savefig(r"../Plots/Bistatic_RCS.pdf")
    return


@app.cell
def _(Onion, bistatic_rcs, np, plt, tw, golden):
    _configs = [
        dict(label='Config A', r=np.array([3e-3,  4e-3, 12e-3]),
             eps=np.array([50, 1, 4]), f0=10e9),
        dict(label='Config B', r=np.array([20e-3, 27e-3, 81e-3]),
             eps=np.array([50, 1, 4]), f0=1.5e9),
    ]

    _colors  = ['steelblue', 'darkorange']
    _BW_frac = 0.3
    _N_freqs = 400

    _fig, _axes = plt.subplots(1, 2, figsize=(tw, tw / golden))

    for _ax, _cfg, _color in zip(_axes, _configs, _colors):
        _f0  = _cfg['f0']
        _r   = _cfg['r']
        _eps = _cfg['eps']
        _r3  = _r[-1]

        _BW     = _BW_frac * _f0
        _f_lo   = _f0 - _BW / 2
        _f_hi   = _f0 + _BW / 2
        _freqs  = np.linspace(_f_lo, _f_hi, _N_freqs)
        _x_axis = (_freqs - _f0) / _BW

        _rcs_fwd_xz = np.zeros(_N_freqs)
        _rcs_fwd_yz = np.zeros(_N_freqs)

        for _i, _freq in enumerate(_freqs):
            _sphere = Onion.from_arrays(_r, _eps, frequency=_freq)
            _, _b_TM, _, _b_TE = _sphere.solve(num_modes=50)
            _b_sc = _b_TM[:, -1]
            _d_sc = _b_TE[:, -1]
            _k    = _sphere.k(-1)
            _rcs_fwd_xz[_i] = bistatic_rcs(_k, 0.0, 0.0,       _b_sc, _d_sc)
            _rcs_fwd_yz[_i] = bistatic_rcs(_k, 0.0, np.pi / 2, _b_sc, _d_sc)

        _norm      = np.pi * _r3**2
        _rcs_xz_dB = 10 * np.log10(np.maximum(_rcs_fwd_xz / _norm, 1e-20))
        _rcs_yz_dB = 10 * np.log10(np.maximum(_rcs_fwd_yz / _norm, 1e-20))

        _peak_idx   = np.argmax(_rcs_xz_dB)
        _peak_x     = _x_axis[_peak_idx]
        _peak_y     = _rcs_xz_dB[_peak_idx]
        _peak_f_GHz = _freqs[_peak_idx] / 1e9

        _ax.plot(_x_axis, _rcs_xz_dB, color=_color,
                 label=r'$\sigma_{3D}(\theta=0^\circ,\;\phi=0^\circ)$ -- XZ (E-plane)')
        _ax.plot(_x_axis, _rcs_yz_dB, color=_color, linestyle='--',
                 label=r'$\sigma_{3D}(\theta=0^\circ,\;\phi=90^\circ)$ -- YZ (H-plane)')

        _ax.plot(_peak_x, _peak_y, marker='*', markersize=8,
                 color='crimson', zorder=5,
                 label=rf'Peak: ${_peak_f_GHz:.3f}$ GHz')

        _ax.axvline(0.0, color='k', linestyle=':', linewidth=0.8,
                    label=rf'$f_0 = {_f0/1e9:.2f}$ GHz')
        _ax.axvspan(-0.5, 0.5, alpha=0.05, color=_color)

        _ax_top = _ax.twiny()
        _ax_top.set_xlim(_f_lo / 1e9, _f_hi / 1e9)
        _ax_top.set_xlabel(r'$f$  (GHz)')
        _tick_freqs = np.linspace(_f_lo, _f_hi, 5)
        _ax_top.set_xticks(_tick_freqs / 1e9)
        _ax_top.set_xticklabels([f'{f/1e9:.3f}' for f in _tick_freqs])

        _layer_str = ',   '.join(
            rf'$\varepsilon_r={e}$, $r={ri*1e3:.0f}$ mm'
            for e, ri in zip(_eps, _r)
        )
        _ax.set_xlabel(r'$(f - f_0)\;/\;\mathrm{BW}$')
        _ax.set_ylabel(r'$10\log_{10}\!\left(\sigma_\mathrm{3D}\,/\,\pi r_3^2\right)$  (dB)')
        _ax.set_title(f'{_cfg["label"]}:  {_layer_str}')
        _ax.set_xlim(-0.5, 0.5)
        _ax.set_xticks(np.linspace(-0.5, 0.5, 11))
        _ax.legend(loc='lower left')

    _fig.suptitle(
        rf'Forward Scattering ($\theta=0^\circ$) -- XZ ($\phi=0^\circ$) vs YZ ($\phi=90^\circ$)'
        rf' -- BW = {_BW_frac*100:.0f}\% of $f_0$',
        fontweight='bold',
    )
    plt.tight_layout()
    _fig.savefig(r"../Plots/Bistatic_RCS_forward_BW.pdf")
    return


@app.cell
def _(Onion, np):
    _r_a = np.array([3e-3, 4e-3, 12e-3])
    _eps_a = np.array([50, 1, 4])
    _sphere_a = Onion.from_arrays(_r_a, _eps_a, frequency=9.816e9)

    _r_b = np.array([20e-3, 27e-3, 81e-3])
    _eps_b = np.array([50, 1, 4])
    _sphere_b = Onion.from_arrays(_r_b, _eps_b, frequency=1.473e9)

    _fig_A = _sphere_a.solve_and_plot_rcs_nearfield(num_modes=50, label="Config A")
    _fig_B = _sphere_b.solve_and_plot_rcs_nearfield(num_modes=50, label="Config B")
    _fig_A.savefig(r"../Plots/Config_A_RCS_nearfield.pdf")
    _fig_B.savefig(r"../Plots/Config_B_RCS_nearfield.pdf")
    return


if __name__ == "__main__":
    app.run()
