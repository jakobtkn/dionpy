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

    return C_0, Onion, bistatic_rcs, mo, np, plt


@app.cell
def _():

    # r_a = np.array([3e-3,4e-3,12e-3])
    # eps_a = np.array([50,1,4])
    # test_a = Onion.from_arrays(r_a, eps_a, frequency=10e9)

    # r_b = np.array([20e-3,27e-3,81e-3])
    # eps_b = np.array([50,1,4])
    # test_b = Onion.from_arrays(r_b, eps_b, frequency=1.5e9)

    # r_c = np.array([20e-3,27e-3,81e-3,27e-3,81e-3,27e-3])
    # eps_c = np.array([50,1,4,4,4,4])
    # test_c = Onion.from_arrays(r_c, eps_c, frequency=1.5e9)
    return


@app.cell
def _():
    # A_TM, A_TE, b, _ = test_a.assemble_one_mode(5)
    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # axes[0].matshow(abs(A_TM) > 0)
    # axes[0].set_title("A_TM")
    # axes[1].matshow(abs(A_TE) > 0)
    # axes[1].set_title("A_TE")
    # axes[2].matshow(abs(b[:,None]) > 0)
    # axes[2].set_title("b")
    # fig
    return


@app.cell
def _():
    # a_TM, b_TM, a_TE, b_TE = test_c.solve(10)
    # test_a.solve_and_plot(num_modes=20)

    # freq = 400e9
    # test_7_13 = Onion.from_arrays(np.array([C_0/freq]), np.array([2.56]),freq)
    # test_7_13.solve_and_plot(num_modes=20)
    # # From 
    # freq = 400e9
    # test_transparent = Onion.from_arrays(np.array([C_0/freq]),np.array([2.56]),frequency=freq)
    # test_transparent.solve_and_plot(num_modes=20)
    return


@app.cell
def _():


    # freq = 400e6
    # test_7_13 = Onion.from_arrays(np.array([C_0/freq]), np.array([2.56]),freq)

    # test_7_13.solve_and_plot(num_modes=20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Example 7_14 and 7_15 from Cheng
    """)
    return


@app.cell
def _():
    # from dionpy.field import scattered_field, scattered_field_xyz

    # freq = 400e9
    # radii_in_lambda = [1.0, 0.5, 0.2]
    # colors          = ['steelblue', 'darkorange', 'seagreen']
    # theta           = np.linspace(0, np.pi, 361)
    # theta_deg       = np.degrees(theta)

    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # for r_lambda, color in zip(radii_in_lambda, colors):

    #     # --- Build sphere whose radius = r_lambda × λ ---
    #     lam0   = C_0 / freq                        # free-space wavelength
    #     radius = r_lambda * lam0                   # physical radius [m]

    #     sphere = Onion.from_arrays(np.array([radius]), np.array([2.56]), freq)
    #     a_TM, b_TM, a_TE, b_TE = sphere.solve(num_modes=30)

    #     b_sc = b_TM[:, -1]                         # TM scattered coefficients
    #     d_sc = b_TE[:, -1]                         # TE scattered coefficients
    #     k    = sphere.k(-1)                        # outer wavenumber
    #     lam  = 2 * np.pi / k                       # wavelength in outer medium

    #     label = f'r = {r_lambda}λ'

    #     # --- E-plane (φ = 0) ---
    #     rcs_E = np.array([bistatic_rcs(k, t, 0,          b_sc, d_sc) for t in theta])
    #     rcs_E_dB = 10 * np.log10(np.maximum(rcs_E / lam**2, 1e-20))
    #     axes[0].plot(theta_deg, rcs_E_dB, color=color, linewidth=1.5, label=label)

    #     # --- H-plane (φ = π/2) ---
    #     rcs_H = np.array([bistatic_rcs(k, t, np.pi/2,   b_sc, d_sc) for t in theta])
    #     rcs_H_dB = 10 * np.log10(np.maximum(rcs_H / lam**2, 1e-20))
    #     axes[1].plot(theta_deg, rcs_H_dB, color=color, linewidth=1.5, label=label)

    # # --- Shared axes formatting ---
    # for ax, plane in zip(axes, ['E-Plane  (φ = 0°)', 'H-Plane  (φ = 90°)']):
    #     ax.set_xlabel('θ (degrees)')
    #     ax.set_ylabel('RCS / λ²  (dB)')
    #     ax.set_title(f'Bistatic RCS vs θ — {plane}')
    #     ax.set_xlim(0, 180)
    #     ax.set_ylim(-50, 25)
    #     ax.set_xticks(np.arange(0, 181, 30))
    #     ax.legend()
    #     ax.grid(True, linestyle=':', alpha=0.6)

    # fig.suptitle('Bistatic Radar Cross Section  (εᵣ = 2.56,  f = 400 GHz)',
    #              fontsize=13, fontweight='bold', y=1.01)
    # plt.tight_layout()
    # fig.savefig(r"C:\Users\holge\git\dionpy\Plots/RCS_example_fig_7_14.png")

    # plt.show()
    return


@app.cell
def _():
    # # ── Monostatic RCS vs r/λ  (Figure 7.15 reproduction) ──────────────────────
    # freq   = 400e9
    # lam0   = C_0 / freq
    # eps_r  = 2.56

    # r_over_lam = np.linspace(0.01, 2.0, 400)   # sweep sphere size
    # rcs_mono   = np.zeros(len(r_over_lam))

    # for i, al in enumerate(r_over_lam):
    #     a      = al * lam0                                      # physical radius
    #     sphere = Onion.from_arrays(np.array([a]), np.array([eps_r]), freq)
    #     a_TM, b_TM, a_TE, b_TE = sphere.solve(num_modes=50)

    #     b_sc = b_TM[:, -1]
    #     d_sc = b_TE[:, -1]
    #     k    = sphere.k(-1)

    #     # backscatter: θ = π, φ = 0
    #     rcs_mono[i] = bistatic_rcs(k, np.pi, 0.0, b_sc, d_sc)

    # # normalise by πa²
    # a_phys      = r_over_lam * lam0
    # rcs_norm    = rcs_mono / (np.pi * a_phys**2)
    # rcs_norm_dB = 10 * np.log10(np.maximum(rcs_norm, 1e-20))

    # # ── Plot ─────────────────────────────────────────────────────────────────────
    # fig, ax = plt.subplots(figsize=(8, 5))

    # ax.plot(r_over_lam, rcs_norm_dB, color='k', linewidth=1.5)

    # ax.set_xlabel('$r/\\lambda$', fontsize=12)
    # ax.set_ylabel('$\\sigma_{3D} / \\pi r^2$  (dB)', fontsize=12)
    # ax.set_title('Monostatic RCS — Dielectric Sphere  ($\\varepsilon_r = 2.56$)', fontsize=12)
    # ax.set_xlim(0, 2.0)
    # ax.set_ylim(-50, 20)
    # ax.set_xticks(np.arange(0, 2.1, 0.5))
    # ax.set_yticks(np.arange(-50, 21, 10))
    # ax.annotate('Rayleigh scattering region', xy=(0.02, -30),
    #             xytext=(0.25, -30), fontsize=10,
    #             arrowprops=dict(arrowstyle='<-', color='k'))
    # ax.grid(True, linestyle=':', alpha=0.6)
    # fig.savefig(r"C:\Users\holge\git\dionpy\Plots/RCS_example_fig_7_15.png")
    # plt.tight_layout()
    # plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Monostatic RCS of PEC (Figure 11-29 balanais)
    """)
    return


@app.cell
def _(C_0, Onion, bistatic_rcs, np, plt):
    # ── Monostatic RCS vs r/λ  (PEC) ──────────────────────
    freq   = 400e9
    lam0   = C_0 / freq
    eps_r  = 100000 

    r_over_lam = np.linspace(0.01, 2.0, 400)   # sweep sphere size
    rcs_mono   = np.zeros(len(r_over_lam))

    for i, al in enumerate(r_over_lam):
        a      = al * lam0                                      # physical radius
        sphere = Onion.from_arrays(np.array([a]), np.array([eps_r]), freq)
        a_TM, b_TM, a_TE, b_TE = sphere.solve(num_modes=50)

        b_sc = b_TM[:, -1]
        d_sc = b_TE[:, -1]
        k    = sphere.k(-1)

        # backscatter: θ = π, φ = 0
        rcs_mono[i] = bistatic_rcs(k, np.pi, 0.0, b_sc, d_sc)

    # normalise by πa²
    a_phys      = r_over_lam * lam0
    rcs_norm    = rcs_mono / (np.pi * a_phys**2)
    rcs_norm_dB = 10 * np.log10(np.maximum(rcs_norm, 1e-20))

    # ── Plot ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(r_over_lam, rcs_norm_dB, color='k', linewidth=1.5)

    ax.set_xlabel('$r/\\lambda$', fontsize=12)
    ax.set_ylabel('$\\sigma_{3D} / \\pi r^2$  (dB)', fontsize=12)
    ax.set_title('Monostatic RCS — PEC Sphere', fontsize=12)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-50, 20)
    ax.set_xticks(np.arange(0, 2.1, 0.5))
    ax.set_yticks(np.arange(-50, 21, 10))
    ax.annotate('Rayleigh scattering region', xy=(0.02, -30),
                xytext=(0.25, -30), fontsize=10,
                arrowprops=dict(arrowstyle='<-', color='k'))
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.savefig(r"C:\Users\holge\git\dionpy\Plots/mono_RCS_PEC.png")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Exercise from Project 2 description
    1. Plot of Bistatic RCS as a function of theta in the XZ and YZ plane
    2. Plot of Forwards RCS as a function of frequency (XZ and YZ plane)
    3. Plot of 3D RCS ($10log10\sigma_{3D}/(𝜋𝑟^2)$) in XZ, YZ and XY plane as well as near field 20log10|E|
    """)
    return


@app.cell
def _():
    # configs = [
    #     dict(label='Config A', r=np.array([3e-3,  4e-3, 12e-3]),
    #          eps=np.array([50, 1, 4]), f0=10e9),
    #     dict(label='Config B', r=np.array([20e-3, 27e-3, 81e-3]),
    #          eps=np.array([50, 1, 4]), f0=1.5e9),
    # ]

    # colors    = ['steelblue', 'darkorange']
    # theta     = np.linspace(0, np.pi, 361)
    # theta_deg = np.degrees(theta)

    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # for cfg, color in zip(configs, colors):
    #     r   = cfg['r']
    #     eps = cfg['eps']
    #     f0  = cfg['f0']
    #     r3  = r[-1]                  # outer radius for normalisation

    #     # --- Build layered sphere from config ---
    #     sphere = Onion.from_arrays(r, eps, f0)
    #     a_TM, b_TM, a_TE, b_TE = sphere.solve(num_modes=30)

    #     b_sc = b_TM[:, -1]          # TM scattered coefficients
    #     d_sc = b_TE[:, -1]          # TE scattered coefficients
    #     k    = sphere.k(-1)         # wavenumber in outer medium

    #     norm = np.pi * r3**2        # normalisation: π r3²

    #     # --- E-plane (φ = 0°) ---
    #     rcs_E    = np.array([bistatic_rcs(k, t, 0,        b_sc, d_sc) for t in theta])
    #     rcs_E_dB = 10 * np.log10(np.maximum(rcs_E / norm, 1e-20))
    #     axes[0].plot(theta_deg, rcs_E_dB, color=color, linewidth=1.5, label=cfg['label'])

    #     # --- H-plane (φ = 90°) ---
    #     rcs_H    = np.array([bistatic_rcs(k, t, np.pi/2,  b_sc, d_sc) for t in theta])
    #     rcs_H_dB = 10 * np.log10(np.maximum(rcs_H / norm, 1e-20))
    #     axes[1].plot(theta_deg, rcs_H_dB, color=color, linewidth=1.5, label=cfg['label'])

    # # --- Shared formatting ---
    # for ax, plane in zip(axes, ['E-Plane  (φ = 0°)', 'H-Plane  (φ = 90°)']):
    #     ax.set_xlabel('θ  (degrees)')
    #     ax.set_ylabel(r'$10\log_{10}\!\left(\sigma_\mathrm{3D}\,/\,\pi r_3^2\right)$  (dB)')
    #     ax.set_title(f'Bistatic RCS — {plane}')
    #     ax.set_xlim(0, 180)
    #     ax.set_xticks(np.arange(0, 181, 30))
    #     ax.legend()
    #     ax.grid(True, linestyle=':', alpha=0.6)

    # fig.suptitle(
    #     r'Bistatic RCS — Config A '
    #     r' vs Config B ',
    #     fontsize=12, fontweight='bold'
    # )
    # plt.tight_layout()
    # plt.savefig(r"C:\Users\holge\git\dionpy\Plots/Bistatic_RCS.png", dpi=150, bbox_inches='tight')
    # plt.show()
    return


@app.cell
def _():
    # configs = [
    #     dict(label='Config A', r=np.array([3e-3,  4e-3, 12e-3]),
    #          eps=np.array([50, 1, 4]), f0=10e9),
    #     dict(label='Config B', r=np.array([20e-3, 27e-3, 81e-3]),
    #          eps=np.array([50, 1, 4]), f0=1.5e9),
    # ]

    # colors  = ['steelblue', 'darkorange']
    # BW_frac = 0.3
    # N_freqs = 400

    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # for ax, cfg, color in zip(axes, configs, colors):
    #     f0  = cfg['f0']
    #     r   = cfg['r']
    #     eps = cfg['eps']
    #     r3  = r[-1]

    #     BW     = BW_frac * f0
    #     f_lo   = f0 - BW / 2
    #     f_hi   = f0 + BW / 2
    #     freqs  = np.linspace(f_lo, f_hi, N_freqs)
    #     x_axis = (freqs - f0) / BW          # normalised x: –0.5 … +0.5

    #     rcs_fwd_xz = np.zeros(N_freqs)      # θ = 0, φ = 0    (XZ / E-plane)
    #     rcs_fwd_yz = np.zeros(N_freqs)      # θ = 0, φ = π/2  (YZ / H-plane)

    #     for i, freq in enumerate(freqs):
    #         sphere = Onion.from_arrays(r, eps, frequency=freq)
    #         _, b_TM, _, b_TE = sphere.solve(num_modes=50)

    #         b_sc = b_TM[:, -1]
    #         d_sc = b_TE[:, -1]
    #         k    = sphere.k(-1)

    #         rcs_fwd_xz[i] = bistatic_rcs(k, 0.0, 0.0,       b_sc, d_sc)  # φ = 0
    #         rcs_fwd_yz[i] = bistatic_rcs(k, 0.0, np.pi / 2, b_sc, d_sc)  # φ = π/2

    #     norm = np.pi * r3**2
    #     rcs_xz_dB = 10 * np.log10(np.maximum(rcs_fwd_xz / norm, 1e-20))
    #     rcs_yz_dB = 10 * np.log10(np.maximum(rcs_fwd_yz / norm, 1e-20))

    #     # ── Peak of XZ forward scattering ────────────────────────────────────────
    #     peak_idx   = np.argmax(rcs_xz_dB)
    #     peak_x     = x_axis[peak_idx]
    #     peak_y     = rcs_xz_dB[peak_idx]
    #     peak_f_GHz = freqs[peak_idx] / 1e9

    #     # ── Plot ─────────────────────────────────────────────────────────────────
    #     ax.plot(x_axis, rcs_xz_dB, color=color, linewidth=1.5,
    #             label=r'$\sigma_{3D}(\theta=0°,\;\phi=0°)$ — XZ (E-plane)')
    #     ax.plot(x_axis, rcs_yz_dB, color=color, linewidth=1.5, linestyle='--',
    #             label=r'$\sigma_{3D}(\theta=0°,\;\phi=90°)$ — YZ (H-plane)')

    #     # Peak marker on XZ curve
    #     ax.plot(peak_x, peak_y, marker='*', markersize=12,
    #             color='crimson', zorder=5,
    #             label=rf'Peak: ${peak_f_GHz:.3f}$ GHz')

    #     # f0 centre line
    #     ax.axvline(0.0, color='k', linestyle=':', linewidth=1.2,
    #                label=rf'$f_0 = {f0/1e9:.2f}$ GHz')
    #     ax.axvspan(-0.5, 0.5, alpha=0.05, color=color)

    #     # Secondary absolute-frequency axis on top
    #     ax_top = ax.twiny()
    #     ax_top.set_xlim(f_lo / 1e9, f_hi / 1e9)
    #     ax_top.set_xlabel(r'$f$  (GHz)', fontsize=10)
    #     tick_freqs = np.linspace(f_lo, f_hi, 5)
    #     ax_top.set_xticks(tick_freqs / 1e9)
    #     ax_top.set_xticklabels([f'{f/1e9:.3f}' for f in tick_freqs], fontsize=8)

    #     layer_str = ',   '.join(
    #         rf'$\varepsilon_r={e}$, $r={ri*1e3:.0f}$ mm'
    #         for e, ri in zip(eps, r)
    #     )
    #     ax.set_xlabel(r'$(f - f_0)\;/\;\mathrm{BW}$', fontsize=11)
    #     ax.set_ylabel(
    #         r'$10\log_{10}\!\left(\sigma_\mathrm{3D}\,/\,\pi r_3^2\right)$  (dB)',
    #         fontsize=11,
    #     )
    #     ax.set_title(f'{cfg["label"]}:  {layer_str}', fontsize=9)
    #     ax.set_xlim(-0.5, 0.5)
    #     ax.set_xticks(np.linspace(-0.5, 0.5, 11))
    #     ax.legend(fontsize=8, loc='lower left')
    #     ax.grid(True, linestyle=':', alpha=0.6)

    # fig.suptitle(
    #     rf'Forward Scattering ($\theta=0°$) — XZ ($\phi=0°$) vs YZ ($\phi=90°$)'
    #     rf' — BW = {BW_frac*100:.0f}% of $f_0$',
    #     fontsize=12, fontweight='bold',
    # )
    # plt.tight_layout()
    # plt.savefig("Bistatic_RCS_forward_BW.png", dpi=150, bbox_inches='tight')
    # plt.show()
    return


@app.cell
def _():
    # r_a = np.array([3e-3,4e-3,12e-3])
    # eps_a = np.array([50,1,4])
    # sphere_a = Onion.from_arrays(r_a, eps_a, frequency=9.816e9)

    # r_b = np.array([20e-3,27e-3,81e-3])
    # eps_b = np.array([50,1,4])
    # sphere_b = Onion.from_arrays(r_b, eps_b, frequency=1.473e9)

    # fig_A = sphere_a.solve_and_plot_rcs_nearfield(num_modes=50, label="Config A")
    # fig_B = sphere_b.solve_and_plot_rcs_nearfield(num_modes=50, label="Config B")
    # fig_A.savefig("Config_A_RCS_nearfield.png", dpi=150, bbox_inches='tight')
    # fig_B.savefig("Config_B_RCS_nearfield.png", dpi=150, bbox_inches='tight')
    return


if __name__ == "__main__":
    app.run()
