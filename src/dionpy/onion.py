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
