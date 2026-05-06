"""Regenerate all publication-quality PDF figures."""
from dionpy.constants import C_0
from dionpy.onion import Onion
from dionpy.plotting import (
    plot_bistatic_rcs,
    plot_monostatic_vs_r,
    plot_forward_bw,
    plot_rcs_nearfield,
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


plt.style.use("../Plots/paper.mplstyle")

OUT = "../Plots"

# ---------------------------------------------------------------------------
# Jin test case — single-layer dielectric sphere, eps_r = 2.56, f = 400 GHz
# Recreates Figure 7.14 and 7.15 from J.-M. Jin,
# "Theory and Computation of Electromagnetic Fields"
# ---------------------------------------------------------------------------
_freq = 400e9
_lam0 = C_0 / _freq
_eps_r = 2.56
_radii = [1.0, 0.5, 0.2]   # r / lambda

jin_configs = [
    dict(label=rf"$r = {r}\lambda$",
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
fig.savefig(f"{OUT}/RCS_example_fig_7_15.pdf")
plt.close(fig)

# ---------------------------------------------------------------------------
# Config A and Config B — layered spheres
# ---------------------------------------------------------------------------
configs = [
    dict(label="Config A", r=np.array(
        [3e-3,  4e-3,  12e-3]), eps=np.array([50, 1, 4]), freq=10e9),
    dict(label="Config B", r=np.array(
        [20e-3, 27e-3, 81e-3]), eps=np.array([50, 1, 4]), freq=1.5e9),
]

# Forward scattering vs frequency — Config A & B combined
print("Generating Bistatic_RCS_forward_BW.pdf ...")
fig, peak_freqs = plot_forward_bw(configs, bw_frac=0.3, n_freqs=400, num_modes=50)
fig.savefig(f"{OUT}/Bistatic_RCS_forward_BW.pdf")
plt.close(fig)

# RCS + Near-field — Config A (at resonance frequency)
print("Generating Config_A_RCS_and_nearfield.pdf ...")
sphere_a = Onion.from_arrays(np.array([3e-3, 4e-3, 12e-3]),
                             np.array([50, 1, 4]), frequency=peak_freqs[0])
fig = plot_rcs_nearfield(sphere_a, num_modes=50, label="Config A")
fig.savefig(f"{OUT}/Config_A_RCS_and_nearfield.pdf")
plt.close(fig)

# RCS + Near-field — Config B (at resonance frequency)
print("Generating Config_B_RCS_and_nearfield.pdf ...")
sphere_b = Onion.from_arrays(np.array([20e-3, 27e-3, 81e-3]),
                             np.array([50, 1, 4]), frequency=peak_freqs[1])
fig = plot_rcs_nearfield(sphere_b, num_modes=50, label="Config B")
fig.savefig(f"{OUT}/Config_B_RCS_and_nearfield.pdf")
plt.close(fig)

print("Done. All PDFs written to ../Plots/")
