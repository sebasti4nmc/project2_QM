import numpy as np
from TISE_rk2 import TISE
import matplotlib.pyplot as plt
from scipy.optimize import brentq

L = 30
E_min, E_max, dE = 0.1, 10.0, 0.1

def shoot(E, L=30):
    if L is None:
            L = E + 3
    tise = TISE(dx=0.0001, e=E)

    out = tise.predict(L)
    x = np.array([p[0] for p in out])
    psi = np.array([p[1] for p in out])

    return psi[len(psi) - 1]   

def main():
    # 1. Coarse scan: evaluate psi(L) on an energy grid
    energies = np.arange(E_min, E_max + dE, dE)
    psi_L = np.array([shoot(E, L=L)for E in energies])

    # 2. Find intervals where psi(L) changes sign
    brackets = [(energies[i], energies[i + 1])
                for i in range(len(energies) - 1)
                if psi_L[i] * psi_L[i + 1] < 0]
    print("Sign changes in:", brackets)

    # 3. Refine each bracket with Brent's method
    roots = []
    for a, b in brackets:
        root, info = brentq(shoot, a, b, xtol=1e-10, full_output=True)
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
