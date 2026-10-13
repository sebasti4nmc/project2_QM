import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit


# ---------------------------------------------------------------
# Jitted numeric core (plain numbers/arrays only, no `self`)
# ---------------------------------------------------------------
@njit
def v(x):
    """Potential: hard wall at x < 0, linear (v = x) for x >= 0."""
    if x < 0:
        return 1e9          # float on both branches so numba can unify types
    return x * 1.0


@njit
def f(x, psi, u):
    """dpsi/dx = u"""
    return u


@njit
def g(E, x, psi, u):
    """du/dx = 2 * psi * (V(x) - E)"""
    return 2.0 * psi * (v(x) - E)


@njit
def rk4_solve(E, dx, L, x0=0.0, psi0=0.0, u0=1.0):
    """
    Integrate the TISE with RK4 from x0 to ~x0+L.
    Returns an (n+1, 3) array with columns [x, psi, u].
    """
    n = math.ceil(L / dx)
    out = np.empty((n + 1, 3))
    x, psi, u = x0, psi0, u0
    out[0, 0] = x
    out[0, 1] = psi
    out[0, 2] = u

    for i in range(1, n + 1):
        # k = steps for psi, l = steps for u
        k1 = dx * f(x, psi, u)
        l1 = dx * g(E, x, psi, u)

        k2 = dx * f(x + dx / 2, psi + k1 / 2, u + l1 / 2)
        l2 = dx * g(E, x + dx / 2, psi + k1 / 2, u + l1 / 2)

        k3 = dx * f(x + dx / 2, psi + k2 / 2, u + l2 / 2)
        l3 = dx * g(E, x + dx / 2, psi + k2 / 2, u + l2 / 2)

        k4 = dx * f(x + dx, psi + k3, u + l3)
        l4 = dx * g(E, x + dx, psi + k3, u + l3)

        psi += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        u += (l1 + 2 * l2 + 2 * l3 + l4) / 6
        x += dx

        out[i, 0] = x
        out[i, 1] = psi
        out[i, 2] = u

    return out


@njit
def psi_at_L(E, dx, L):
    """Final psi value at x = L (what the shooting method needs)."""
    return rk4_solve(E, dx, L)[-1, 1]


@njit
def scan_energies(energies, dx, L):
    """psi(L) for every energy in an array, all inside compiled code."""
    res = np.empty(len(energies))
    for i in range(len(energies)):
        res[i] = psi_at_L(energies[i], dx, L)
    return res


# ---------------------------------------------------------------
# Thin (non-jitted) wrapper class, kept for API compatibility
# ---------------------------------------------------------------
class TISE:
    def __init__(self, dx, e, psi_0=0.0, u0=1.0, x0=0.0):
        self.dx = dx
        self.e = e
        self.x = x0
        self.psi = psi_0
        self.u = u0

    def predict(self, L):
        """Returns an (n+1, 3) array: columns x, psi, u."""
        return rk4_solve(self.e, self.dx, L, self.x, self.psi, self.u)


# ---------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------
def plot_eq(eq, E, L=None, pad=1.5):
    if L is None:
        L = E + 3               # a little past the turning point

    out = eq.predict(L)
    x = out[:, 0]
    psi = out[:, 1]

    allowed = x <= E            # oscillatory region
    amp = np.max(np.abs(psi[allowed]))

    plt.plot(x, psi)
    plt.ylim(-pad * amp, pad * amp)        # clip the divergent tail
    plt.axvline(E, ls=":", color="gray")   # turning point
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("psi")