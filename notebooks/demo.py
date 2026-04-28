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
    from dionpy.constants import C_0
    import numpy as np
    import matplotlib.pyplot as plt

    return C_0, Onion, np, plt


@app.cell
def _(Onion, np):

    r_a = np.array([3e-3,4e-3,12e-3])
    eps_a = np.array([50,1,4])
    test_a = Onion.from_arrays(r_a, eps_a, frequency=10e9)

    r_b = np.array([20e-3,27e-3,81e-3])
    eps_b = np.array([50,1,4])
    test_b = Onion.from_arrays(r_b, eps_b, frequency=1.5e9)

    r_c = np.array([20e-3,27e-3,81e-3,27e-3,81e-3,27e-3])
    eps_c = np.array([50,1,4,4,4,4])
    test_c = Onion.from_arrays(r_c, eps_c, frequency=1.5e9)
    return test_a, test_c


@app.cell
def _(plt, test_a):
    A_TM, A_TE, b, _ = test_a.assemble_one_mode(5)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].matshow(abs(A_TM) > 0)
    axes[0].set_title("A_TM")
    axes[1].matshow(abs(A_TE) > 0)
    axes[1].set_title("A_TE")
    axes[2].matshow(abs(b[:,None]) > 0)
    axes[2].set_title("b")
    fig
    return


@app.cell
def _(test_c):
    a_TM, b_TM, a_TE, b_TE = test_c.solve(10)
    return


@app.cell
def _(C_0, Onion, np):
    freq = 400e6
    test_7_13 = Onion.from_arrays(np.array([C_0/freq]), np.array([2.56]),freq)

    fig = test_7_13.solve_and_plot(num_modes=20)
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
