import numpy as np

# Permittivity of free space [F/m]
EPS_0 = 8.8541878128e-12

# Permeability of free space [H/m]
MU_0 = 1.25663706212e-6

# Speed of light in vacuum [m/s]
C_0 = 1 / np.sqrt(MU_0 * EPS_0)

# Free-space wave impedance [Ω]
ETA_0 = np.sqrt(MU_0 / EPS_0)

