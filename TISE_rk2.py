import math
import numpy as np
import matplotlib.pyplot as plt

E = 1

def v(x):
    if x < 0:
        return 1000000000
    if x >=0:
        return x


def f(u, x=0, psi=0): 
    return u

def g(x, psi, u):
    return -2 * psi * (v(x) - E)

class ODE:
    def __init__(self, u_eq, psi_eq, dt, u0, psi_0, t0=0):
        self.f = psi_eq
        self.g = u_eq
        self.t = t0
        self.u = u0
        self.psi = psi_0
        self.dt = dt
        pass

    def getNext(self):
        t, u, psi, dt = self.t, self.u, self.psi, self.dt

        # Compute k1 and k2 - steps for u
        # k1 = dt * self.f(
        # k2 = dx * self.f(x + dx, y + k1)

        # Compute l1 and l2 - steps for psi

        # Compute next step for y
        # self.y += (k1 + k2) / 2
        # self.x += dx

        # return(self.x, self.y)

    def predict(self, interval):
        n = math.ceil(interval / self.dx)
        frames = [(self.x, self.y)]  # include initial condition
        for _ in range(n):
            frames.append(self.getNext())

        return frames