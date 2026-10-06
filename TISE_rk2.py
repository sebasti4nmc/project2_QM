import math
import numpy as np
import matplotlib.pyplot as plt

ENERGY = 3.245

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
    def __init__(self, u_eq, psi_eq, dx, e=ENERGY, psi_0=0, u0=1, x0=0):
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

        # Compute k1 (step for psi) and l1 (step for u)
        k1 = dx * self.f(x, psi, u)
        l1 = dx * self.g(e, x, psi, u)

        # Using k1, l1 - compute k2 and l2
        k2 = dx * self.f(x +dx, psi + k1, u + l1)
        l2 = dx * self.g(e, x +dx, psi + k1, u + l1)

        # Compute next step for y
        self.psi += (k1 + k2) / 2
        self.u += (l1 + l2) / 2
        self.x += dx

        return(self.x, self.psi, self.u)

    def predict(self, L):
        n = math.ceil(L / self.dx)
        frames = [(self.x, self.psi, self.u)]  # include initial condition
        for _ in range(n):
            frames.append(self.getNext())

        return frames

def plot_eq(eq, E, L=None, pad=1.5):
    if L is None:
        L = E + 5                  # a little past the turning point

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
    
def main():
    tise = TISE(u_eq=g, psi_eq=f, dx=0.00001)

    plot_eq(tise, E=ENERGY)

    psi_Ls = []
    # for i in range (90,150):
    #     energy = i
        
    #     tise = TISE(u_eq=g, psi_eq=f, dx=0.01, e=energy)

    #     plot_eq(tise, L=energy+5)
    
    plt.show()
        
        
        

if __name__ == "__main__":
    main()