import math
import numpy as np
import matplotlib.pyplot as plt

def eq_1(x, y): 
    return -x * y # ODE representing our system: dy/dx = -xy

class ODE:
    def __init__(self, ode, y0, dx, x0=0):
        self.f = ode
        self.x = x0
        self.y = y0
        self.dx = dx
        pass

    def getNext(self):
        x, y, dx = self.x, self.y, self.dx

        # Compute k1 and k2 for rk2 steps
        k1 = dx * self.f(x, y)
        k2 = dx * self.f(x + dx, y + k1)

        # Compute next step for y
        self.y += (k1 + k2) / 2
        self.x += dx

        return(self.x, self.y)

    def predict(self, interval):
        n = math.ceil(interval / self.dx)
        frames = [(self.x, self.y)]  # include initial condition
        for _ in range(n):
            frames.append(self.getNext())

        return frames

def plot_eq(ode, x0, y0, interval=10):
    out = ode.predict(interval)
    x = np.array([p[0] for p in out])
    y = np.array([p[1] for p in out])

    exact = y0 * np.exp(-(x**2 - x0**2) / 2)

    plt.plot(x, y, 'o-', label="RK2")
    # plt.plot(x, exact, '--', label="Analytical Solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

def main():
    x0, y0 = 0, 1
    ode = ODE(eq_1, y0=y0, dx=.1, x0=x0)

    plot_eq(ode, x0, y0)

if __name__ == "__main__":
    main()


