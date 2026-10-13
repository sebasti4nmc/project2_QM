import sympy as sp
import numpy as np
from scipy.special import airy, ai_zeros
import matplotlib.pyplot as plt

# def v(x):
#     if x < 0:
#         return 1000000000
#     if x >=0:
#         return x

c = 2 ** (1/3)

def gen_solution():
    x = sp.symbols('x')
    E = sp.symbols('E', positive=True) 
    psi = sp.Function('psi')

    # Define ODE: ψ'' = 2(x - E)ψ
    ode = sp.Eq(psi(x).diff(x, x) - 2 * (x - E) * psi(x), 0)

    # Solve symbolically
    solution = sp.dsolve(ode, psi(x))

    return solution

def eigen_calc(n):
    a_n = ai_zeros(n)[0]            # first n zeros of Ai (negative numbers)
    E_n = -a_n / 2**(1/3)

    return E_n[n - 1]

def coefficients(E):
    z0 = -c * E
    Ai, Aip, Bi, Bip = airy(z0)
    C1 = -np.pi * Bi / c
    C2 =  np.pi * Ai / c
    return C1, C2

def psi_exact(x, E):
    C1, C2 = coefficients(E)
    Ai, _, Bi, _ = airy(c * (x - E))
    return C1 * Ai + C2 * Bi

def plot_exact(E, pad=1.5, label=None):
    x = np.linspace(0, E + 3, 2000)
    psi = psi_exact(x, E)

    amp = np.max(np.abs(psi[x <= E]))
    plt.plot(x, psi, label=label or f"E = {E:.4f}")
    plt.ylim(-pad * amp, pad * amp)
    plt.axvline(E, ls=":", color="gray")      # turning point
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("x")
    plt.ylabel("psi")

def main():
    eigenvalues = [None]
    for i in range(1,10):
        E = eigen_calc(i)
        eigenvalues.append(E)

    plot_exact(eigenvalues[1])        # 4th eigenvalue, about 5.387
    plt.legend()
    plt.show()
        

main()