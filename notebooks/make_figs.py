"""Regenerate all publication-quality PDF figures."""
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dionpy.onion import (
    Onion,
    plot_nearfield,
    plot_bistatic_rcs,
    plot_monostatic_vs_r,
    plot_forward_bw,
)
from dionpy.constants import C_0

plt.style.use("../Plots/paper.mplstyle")

OUT = "../Plots"

# ---------------------------------------------------------------------------
# Jin test case — single-layer dielectric sphere, eps_r = 2.56, f = 400 GHz
# ---------------------------------------------------------------------------
_freq   = 400e9
_lam0   = C_0 / _freq
_eps_r  = 2.56
_radii  = [1.0, 0.5, 0.2]   # r / lambda

jin_configs = [
    dict(label=rf"r = {r}$\lambda$",
         r=np.array([r * _lam0]),
         eps=np.array([_eps_r]),
         freq=_freq)
    for r in _radii
]

# Figure 7.14 — Bistatic RCS vs theta
print("Generating RCS_example_fig_7_14.pdf ...")
fig = plot_bistatic_rcs(
    jin_configs,
    num_modes=30,
    norm="lambda",
    ylim=(-50, 25),
    title=rf"Bistatic Radar Cross Section  ($\varepsilon_r = {_eps_r}$)",
)
fig.savefig(f"{OUT}/RCS_example_fig_7_14.pdf")
plt.close(fig)

# Figure 7.15 — Monostatic RCS vs r / lambda
print("Generating RCS_example_fig_7_15.pdf ...")
fig = plot_monostatic_vs_r(
    permittivities=_eps_r,
    freq=_freq,
    num_modes=50,
    title=rf"Monostatic RCS --- Dielectric Sphere  ($\varepsilon_r = {_eps_r}$)",
)
fig.axes[0].set_ylim(-50, 20)
fig.axes[0].annotate(
    "Rayleigh scattering region", xy=(0.02, -30), xytext=(0.25, -30),
    arrowprops=dict(arrowstyle="<-", color="k"),
)
fig.tight_layout()
fig.savefig(f"{OUT}/RCS_example_fig_7_15.pdf")
plt.close(fig)

# Monostatic RCS — PEC sphere (eps_r → ∞ approximated by large value)
print("Generating mono_RCS_PEC.pdf ...")
fig = plot_monostatic_vs_r(
    permittivities=1e5,
    freq=_freq,
    num_modes=50,
    title="Monostatic RCS --- PEC Sphere",
)
fig.axes[0].set_ylim(-50, 20)
fig.axes[0].annotate(
    "Rayleigh scattering region", xy=(0.02, -30), xytext=(0.25, -30),
    arrowprops=dict(arrowstyle="<-", color="k"),
)
fig.tight_layout()
fig.savefig(f"{OUT}/mono_RCS_PEC.pdf")
plt.close(fig)

# Near-field for r = 0.5 lambda Jin sphere
print("Generating RCS_jin_nearfield.pdf ...")
jin_sphere = Onion.from_arrays(np.array([0.5 * _lam0]), np.array([_eps_r]), _freq)
fig = plot_nearfield(jin_sphere, num_modes=50,
                     title=rf"Near-Field  ($\varepsilon_r = {_eps_r}$,  r = 0.5$\lambda$,  f = 400 GHz)")
fig.savefig(f"{OUT}/RCS_jin_nearfield.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Config A and Config B — layered spheres
# ---------------------------------------------------------------------------
configs = [
    dict(label="Config A", r=np.array([3e-3,  4e-3,  12e-3]), eps=np.array([50, 1, 4]), freq=10e9),
    dict(label="Config B", r=np.array([20e-3, 27e-3, 81e-3]), eps=np.array([50, 1, 4]), freq=1.5e9),
]

# Bistatic RCS — Config A vs Config B
print("Generating Bistatic_RCS.pdf ...")
fig = plot_bistatic_rcs(
    configs,
    num_modes=30,
    norm="r3",
    title="Bistatic RCS -- Config A vs Config B",
)
fig.savefig(f"{OUT}/Bistatic_RCS.pdf")
plt.close(fig)

# Forward scattering vs frequency
print("Generating Bistatic_RCS_forward_BW.pdf ...")
fig = plot_forward_bw(configs, bw_frac=0.3, n_freqs=400, num_modes=50)
fig.savefig(f"{OUT}/Bistatic_RCS_forward_BW.pdf")
plt.close(fig)

# Near-field — Config A (at resonance frequency)
print("Generating Config_A_RCS_nearfield.pdf ...")
sphere_a = Onion.from_arrays(np.array([3e-3, 4e-3, 12e-3]),
                              np.array([50, 1, 4]), frequency=9.816e9)
fig = sphere_a.solve_and_plot_rcs_nearfield(num_modes=50, label="Config A")
fig.savefig(f"{OUT}/Config_A_RCS_nearfield.pdf")
plt.close(fig)

# Near-field — Config B (at resonance frequency)
print("Generating Config_B_RCS_nearfield.pdf ...")
sphere_b = Onion.from_arrays(np.array([20e-3, 27e-3, 81e-3]),
                              np.array([50, 1, 4]), frequency=1.473e9)
fig = sphere_b.solve_and_plot_rcs_nearfield(num_modes=50, label="Config B")
fig.savefig(f"{OUT}/Config_B_RCS_nearfield.pdf")
plt.close(fig)

print("Done. All PDFs written to ../Plots/")
