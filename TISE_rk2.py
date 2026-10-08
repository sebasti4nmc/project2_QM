import math
from multiprocessing.util import info
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def v(x):
    if x < 0:
        return 1000000000
    if x >=0:
        return x

def f(x, psi, u=1): 
    return u

def g(E, x, psi, u=1):
    return 2 * psi * (v(x) - E)

class TISE:
    def __init__(self, dx, e, u_eq=g, psi_eq=f, psi_0=0, u0=1, x0=0):
        self.f = psi_eq
        self.g = u_eq
        self.x = x0
        self.u = u0
        self.psi = psi_0
        self.e = e
        self.dx = dx
        pass

    def getNext(self):
        x, u, psi, dx, e = self.x, self.u, self.psi, self.dx, self.e

        # k = steps for psi, l = steps for u
        k1 = dx * self.f(x, psi, u)
        l1 = dx * self.g(e, x, psi, u)

        k2 = dx * self.f(x + dx/2, psi + k1/2, u + l1/2)
        l2 = dx * self.g(e, x + dx/2, psi + k1/2, u + l1/2)

        k3 = dx * self.f(x + dx/2, psi + k2/2, u + l2/2)
        l3 = dx * self.g(e, x + dx/2, psi + k2/2, u + l2/2)

        k4 = dx * self.f(x + dx, psi + k3, u + l3)
        l4 = dx * self.g(e, x + dx, psi + k3, u + l3)

        self.psi += (k1 + 2*k2 + 2*k3 + k4) / 6
        self.u   += (l1 + 2*l2 + 2*l3 + l4) / 6
        self.x   += dx

        return (self.x, self.psi, self.u)

    def predict(self, L):
        n = math.ceil(L / self.dx)
        frames = [(self.x, self.psi, self.u)]  # include initial condition
        for _ in range(n):
            frames.append(self.getNext())

        return frames

def plot_eq(eq, E, L=None, pad=1.5):
    if L is None:
        L = E + 3                 # a little past the turning point

    out = eq.predict(L)
    x = np.array([p[0] for p in out])
    psi = np.array([p[1] for p in out])

    allowed = x <= E                          # oscillatory region
    amp = np.max(np.abs(psi[allowed]))

    plt.plot(x, psi)
    plt.ylim(-pad * amp, pad * amp)           # clip the divergent tail
    plt.axvline(E, ls=":", color="gray")      # turning point
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel("x")
    plt.ylabel("psi")

def shoot(E, L=None):
    if L is None:
            L = E + 3
    tise = TISE(dx=0.0001, e=E)

    out = tise.predict(L)
    x = np.array([p[0] for p in out])
    psi = np.array([p[1] for p in out])

    return psi[len(psi) - 1]
    
L = 30
E_min, E_max, dE = 0.1, 10.0, 0.1

def psi_at_L(E):
    """psi(L) for a trial energy E."""
    return shoot(E, L=L)          

def main():
    # 1. Coarse scan: evaluate psi(L) on an energy grid
    energies = np.arange(E_min, E_max + dE, dE)
    psi_L = np.array([psi_at_L(E) for E in energies])

    # 2. Find intervals where psi(L) changes sign
    brackets = [(energies[i], energies[i + 1])
                for i in range(len(energies) - 1)
                if psi_L[i] * psi_L[i + 1] < 0]
    print("Sign changes in:", brackets)

    # 3. Refine each bracket with Brent's method
    roots = []
    for a, b in brackets:
        root, info = brentq(psi_at_L, a, b, xtol=1e-10, full_output=True)
        print(f"root = {root:.8f}, iterations = {info.iterations}, converged = {info.converged}")
        roots.append(root)
    print("Eigenvalues:", roots)

    # 4. Plot the scan, with the refined roots marked on the zero line
    plt.plot(energies, psi_L, "o-", label="psi(L)")
    plt.plot(roots, np.zeros(len(roots)), "rx", markersize=10, label="refined roots")
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("E")
    plt.ylabel("psi(L)")
    plt.yscale("symlog")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()