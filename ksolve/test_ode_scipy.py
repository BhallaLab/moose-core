# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from scipy.integrate import ode

def func(t, u, mu=1e4):
    u1 = u[1]
    u2 = mu*(1 - u[0]*u[0])*u[1] - u[0]
    return np.array([u1, u2])

def func2(t, y):
    ydot0 = 1e4 * y[1] * y[2] - 0.04e0 * y[0];
    ydot2 = 3e7 * y[1] * y[1]
    ydot1 = -1 *(ydot0 * ydot2)
    return np.array([ydot0, ydot1, ydot2])


def test1():
    u0 = [1e0, 0, 0]
    t0 = 0.0
    tf = 0.4
    name = 'lsoda'
    solver = ode(func2)
    solver.set_integrator(name)
    solver.set_initial_value(u0, t0)

    tvals = []
    i = 0
    while solver.successful() and solver.t < tf:
        solver.integrate(tf, step=True)
        print( solver.t, solver.y )

def test2():
    u0 = [10, 0]
    t0 = 0.0
    tf = 10
    name = 'lsoda'
    solver = ode(func)
    solver.set_integrator(name)
    solver.set_initial_value(u0, t0)

    tvals = []
    i = 0
    while solver.successful() and solver.t < tf:
        solver.integrate(tf, step=True)
        print( solver.t, solver.y )
    tvals = np.unique(tvals)
    print("len(tvals) =", len(tvals))


def main():
    test1()
    test2()

if __name__ == '__main__':
    main()
