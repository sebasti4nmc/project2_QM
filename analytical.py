import sympy as sp


# def v(x):
#     if x < 0:
#         return 1000000000
#     if x >=0:
#         return x

x = sp.symbols('x')
E = sp.symbols('E', positive=True) 
psi = sp.Function('psi')

# ics = {y(0): 2, y(x).diff(x).subs(x, 0): 1}

# Define ODE: ψ'' = 2(x - E)ψ
ode = sp.Eq(psi(x).diff(x, x) - 2 * (x - E) * psi(x), 0)


# Solve symbolically
solution = sp.dsolve(ode, psi(x))
print(solution)  

# solution = sp.dsolve(ode, psi(x), ics=ics)
# print(solution)
