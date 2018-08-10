# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from scipy.integrate import ode


def func(t, u, mu):
    tvals.append(t)
    u1 = u[1]
    u2 = mu*(1 - u[0]*u[0])*u[1] - u[0]
    return np.array([u1, u2])


def jac(t, u, mu):
    j = np.empty((2, 2))
    j[0, 0] = 0.0
    j[0, 1] = 1.0
    j[1, 0] = -mu*2*u[0]*u[1] - 1
    j[1, 1] = mu*(1 - u[0]*u[0])
    return j


mu = 10000.0
u0 = [2, 0]
t0 = 0.0
tf = 10

for name, kwargs in [ #('vode', dict(method='adams')),
                     #('vode', dict(method='bdf')),
                     ('lsoda', {})]:
    for j in [None, jac]:
        solver = ode(func, jac=j)
        solver.set_integrator(name, atol=1e-8, rtol=1e-6, **kwargs)
        solver.set_f_params(mu)
        solver.set_jac_params(mu)
        solver.set_initial_value(u0, t0)

        tvals = []
        i = 0
        while solver.successful() and solver.t < tf:
            solver.integrate(tf, step=True)
            print( solver.t, solver.y )
            i += 1

        print("%-6s %-8s jac=%-5s " %
              (name, kwargs.get('method', ''), j if j else None),
              end='')

        tvals = np.unique(tvals)
        print("len(tvals) =", len(tvals))
