import math
import numpy as np
import matplotlib.pyplot as plt

E = 1

def v(x):
    if x < 0:
        return 1000000000
    if x >=0:
        return x

def f(x, psi, u=1): 
                return u

def g(x, psi, u=1):
    return 2 * psi * (v(x) - E)

class TISE:
    def __init__(self, u_eq, psi_eq, dx, psi_0=0, u0=1, x0=0):
        self.f = psi_eq
        self.g = u_eq
        self.x = x0
        self.u = u0
        self.psi = psi_0
        self.dx = dx
        pass

    def getNext(self):
        x, u, psi, dx = self.x, self.u, self.psi, self.dx

        # Compute k1 (step for psi) and l1 (step for u)
        k1 = dx * self.f(x, psi, u)
        l1 = dx * self.g(x, psi, u)

        # Using k1, l1 - compute k2 and l2
        k2 = dx * self.f(x +dx, psi + k1, u + l1)
        l2 = dx * self.g(x +dx, psi + k1, u + l1)

        # Compute next step for y
        self.psi += (k1 + k2) / 2
        self.u += (l1 + l2) / 2
        self.x += dx

        return(self.x, self.psi, self.u)

    def predict(self, interval):
        n = math.ceil(interval / self.dx)
        frames = [(self.x, self.psi, self.u)]  # include initial condition
        for _ in range(n):
            frames.append(self.getNext())

        return frames

def plot_eq(eq, interval=10):
    out = eq.predict(interval)
    x = np.array([p[0] for p in out])
    psi = np.array([p[1] for p in out])

    plt.plot(x, psi, label="RK2")
    plt.xlabel("x")
    plt.ylabel("psi")
    plt.legend()
    plt.show()
    
def main():
    tise = TISE(u_eq=g, psi_eq=f, dx=0.1)

    plot_eq(tise)

    for i in range (1,11):
        E = i
        def g(x, psi, u=1):
            return 2 * psi * (v(x) - E)
        
        tise = TISE(u_eq=g, psi_eq=f, dx=0.1)

        out = tise.predict(interval=10)
        psi_L = out[1][len(out - 1)]
        print (psi_L)

        
        
        

if __name__ == "__main__":
    main()