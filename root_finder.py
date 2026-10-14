import numpy as np
from TISE_rk2 import TISE
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import os

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

import analytical 

def plot_states(roots, dx=.0001, pad=3, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)
    exact_energies = []

    for n, E in enumerate(roots, start=1):
        out = np.array(TISE(dx=dx, e=E).predict(E + pad))
        x, psi_num = out[:, 0], out[:, 1]

        E_exact = analytical.eigen_calc(n)
        exact_energies.append(E_exact)
        psi_ex = analytical.psi_exact(x, E_exact)

        # normalize both with the exact solution's norm, so any difference
        # between the curves is real error and not a scale choice
        N = np.sqrt(np.sum(psi_ex**2) * dx)
        psi_num, psi_ex = psi_num / N, psi_ex / N

        amp = np.max(np.abs(psi_ex[x <= E_exact]))

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(x, psi_ex, label=f"exact (E = {E_exact:.6f})")
        ax.plot(x, psi_num, "--", label=f"numerical (E = {E:.6f})")
        ax.axvline(E_exact, ls=":", color="gray")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylim(-1.5 * amp, 1.5 * amp)
        ax.set_xlabel("x")
        ax.set_ylabel("psi")
        ax.set_title(f"state n = {n}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"psi_n{n}.png"), dpi=150)
        plt.close(fig)

    return exact_energies

def main():
    # 1. Coarse scan: evaluate psi(L) on an energy grid
    energies = np.arange(E_min, E_max + dE, dE)
    print("Made it to here\n")
    psi_L = np.array([shoot(E, L=L)for E in energies])
    print("Found psi(L) values\n")

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

    plot_states(roots)
    # plt.show()


if __name__ == "__main__":
    main()
