# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "dionpy @ https://jakobtkn.github.io/dionpy/dionpy-0.1.0-py3-none-any.whl",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from dionpy.onion import Onion
    from dionpy.plotting import (
        plot_nearfield,
        plot_bistatic_rcs,
        plot_monostatic_vs_r,
    )
    return Onion, mo, np, plot_bistatic_rcs, plot_monostatic_vs_r, plot_nearfield


@app.cell
def _(mo):
    is_script = mo.app_meta().mode == "script"
    return (is_script,)


@app.cell
def _(mo):
    config_form = (
        mo.md(r"""
# Onion Sphere — Interactive Scattering Explorer

## Sphere Configuration

| Parameter | Value |
|-----------|-------|
| Frequency (GHz) | {freq_ghz} |
| Number of modes | {num_modes} |
| Outer radii (mm), comma-separated | {radii_str} |
| Permittivities εᵣ, comma-separated | {eps_str} |

{include_mono} Include monostatic RCS sweep *(slow — sweeps r/λ ∈ [0.01, 2])*
        """)
        .batch(
            freq_ghz=mo.ui.number(start=0.1, stop=1000.0, step=0.1, value=10.0),
            num_modes=mo.ui.slider(start=5, stop=100, step=5, value=30),
            radii_str=mo.ui.text(value="3, 4, 12", full_width=True),
            eps_str=mo.ui.text(value="50, 1, 4", full_width=True),
            include_mono=mo.ui.checkbox(value=False),
        )
        .form(submit_button_label="Solve & Plot", clear_on_submit=False)
    )
    config_form
    return (config_form,)


@app.cell
def _(
    Onion,
    config_form,
    is_script,
    mo,
    np,
    plot_bistatic_rcs,
    plot_monostatic_vs_r,
    plot_nearfield,
):
    if is_script:
        _v = {
            "freq_ghz": 10.0,
            "num_modes": 30,
            "radii_str": "3, 4, 12",
            "eps_str": "50, 1, 4",
            "include_mono": False,
        }
    else:
        mo.stop(
            config_form.value is None,
            mo.md("⬆️ Configure the sphere above and click **Solve & Plot**."),
        )
        _v = config_form.value

    _freq = _v["freq_ghz"] * 1e9
    _n_modes = _v["num_modes"]
    _radii = np.array([float(x) for x in _v["radii_str"].split(",")]) * 1e-3
    _eps = np.array([float(x) for x in _v["eps_str"].split(",")])

    _sphere = Onion.from_arrays(_radii, _eps, _freq)

    _fig_bistatic = plot_bistatic_rcs(
        [dict(r=_radii, eps=_eps, freq=_freq, label="Sphere")],
        num_modes=_n_modes,
        norm="r3",
        title=rf"Bistatic RCS  (f = {_v['freq_ghz']:.2f} GHz)",
    )

    _fig_nearfield = plot_nearfield(_sphere, num_modes=_n_modes)

    _fig_combined = _sphere.solve_and_plot_rcs_nearfield(
        num_modes=_n_modes,
        label=rf"f = {_v['freq_ghz']:.2f} GHz",
    )

    _figs = [_fig_bistatic, _fig_nearfield, _fig_combined]

    if _v["include_mono"]:
        _fig_mono = plot_monostatic_vs_r(
            permittivities=_eps,
            freq=_freq,
            r_fractions=_radii / _radii[-1],
            r_over_lam=np.linspace(0.01, 2.0, 100),
            num_modes=_n_modes,
        )
        _figs.append(_fig_mono)

    mo.vstack(_figs)
    return


if __name__ == "__main__":
    app.run()
