import numpy as np
import math
from dataclasses import dataclass, field
from enum import Enum

from .constants import MU_0, EPS_0, ETA_0
from .functions import riccati_h1, riccati_h2


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

    def solve(self, num_modes: int):
        num_layers = len(self.layers)
        a_TM = np.zeros((num_modes, num_layers), dtype=complex)
        b_TM = np.zeros((num_modes, num_layers), dtype=complex)
        a_TE = np.zeros((num_modes, num_layers), dtype=complex)
        b_TE = np.zeros((num_modes, num_layers), dtype=complex)

        for m in range(1, num_modes):
            A_TM, A_TE, rhs_TM, rhs_TE = self.assemble_one_mode(m)
            x_TM = np.linalg.solve(A_TM, rhs_TM)
            x_TE = np.linalg.solve(A_TE, rhs_TE)
            a_TM[m, :] = x_TM[0::2]
            b_TM[m, :] = x_TM[1::2]
            a_TE[m, :] = x_TE[0::2]
            b_TE[m, :] = x_TE[1::2]

        return a_TM, b_TM, a_TE, b_TE

    def assemble_one_mode(self, mode):
        """
        (a0_m,b0_m,a1_m,b1_m,...,al_m,bl_m)
        """
        num_layers: int = len(self.layers)
        num_unknowns: int = 2*num_layers
        A_TM = np.zeros((num_unknowns, num_unknowns), dtype=complex)
        A_TE = np.zeros((num_unknowns, num_unknowns), dtype=complex)
        rhs_TM = np.zeros(num_unknowns, dtype=complex)
        rhs_TE = np.zeros(num_unknowns, dtype=complex)

        xs_in = np.array([self.k(i) * self.r(i)
                         for i in range(num_layers - 1)])
        xs_out = np.array([self.k(i+1) * self.r(i)
                          for i in range(num_layers - 1)])


        # Precompute Riccati Functions
        H1_in,  H1p_in = riccati_h1(mode, xs_in),  riccati_h1(
            mode, xs_in,  deriv=True)
        H2_in,  H2p_in = riccati_h2(mode, xs_in),  riccati_h2(
            mode, xs_in,  deriv=True)
        H1_out, H1p_out = riccati_h1(
            mode, xs_out), riccati_h1(mode, xs_out, deriv=True)
        H2_out, H2p_out = riccati_h2(
            mode, xs_out), riccati_h2(mode, xs_out, deriv=True)

        def bc_E_TE(i):
            return np.array([H1_in[i]/self.mu(i),   H2_in[i]/self.mu(i),
                             -H1_out[i]/self.mu(i+1), -H2_out[i]/self.mu(i+1)])

        def bc_H_TE(i):
            return np.array([H1p_in[i]/self.k(i),   H2p_in[i]/self.k(i),
                             -H1p_out[i]/self.k(i+1), -H2p_out[i]/self.k(i+1)])

        def bc_E_TM(i):
            return np.array([H1_in[i]/self.k(i),   H2_in[i]/self.k(i),
                             -H1_out[i]/self.k(i+1), -H2_out[i]/self.k(i+1)])

        def bc_H_TM(i):
            return np.array([H1p_in[i]/self.mu(i),   H2p_in[i]/self.mu(i),
                             -H1p_out[i]/self.mu(i+1), -H2p_out[i]/self.mu(i+1)])

        # Enforce a^0 = b^0
        A_TM[0, :2] = [1, -1]
        A_TE[0, :2] = [1, -1]

        # Enforce interface conditions
        for layer in self.layers[:-1]:
            i = layer.index
            A_TM[layer.row_E, layer.cols] = bc_E_TM(i)
            A_TM[layer.row_H, layer.cols] = bc_H_TM(i)
            A_TE[layer.row_E, layer.cols] = bc_E_TE(i)
            A_TE[layer.row_H, layer.cols] = bc_H_TE(i)

        # RHS — xs_out[-1] = k(-1)*r(-2), reuse Jn/Jnp derived above
        Jn = (H1_out[-1] + H2_out[-1]) / 2
        Jnp = (H1p_out[-1] + H2p_out[-1]) / 2
        factor = 1j**(-mode)*(2*mode+1)/(mode*(mode+1))
        last_interface_layer: Layer = self.layers[-2]
        rhs_TM[last_interface_layer.row_E] = factor * Jn / self.k(-1)
        rhs_TM[last_interface_layer.row_H] = factor * Jnp / self.mu(-1)
        rhs_TE[last_interface_layer.row_E] = factor * Jn / self.mu(-1)
        rhs_TE[last_interface_layer.row_H] = factor * Jnp / self.k(-1)

        # Enforce a^l = 0
        A_TE[:, -2] = 0
        A_TM[:, -2] = 0
        A_TE[-1, -2] = 1
        A_TM[-1, -2] = 1

        return A_TM, A_TE, rhs_TM, rhs_TE
