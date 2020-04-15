# Modified from function.py ---

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import moose

simtime = 1.0

def test_example():
    moose.Neutral('/model')
    function = moose.Function('/model/function')
    function.c['c0'] = 1.0
    function.c['c1'] = 2.0
    #function.x.num = 1
    function.expr = 'c0 * exp(c1*x0) * cos(y0) + sin(t)'
    # mode 0 - evaluate function value, derivative and rate
    # mode 1 - just evaluate function value,
    # mode 2 - evaluate derivative,
    # mode 3 - evaluate rate
    function.mode = 0
    function.independent = 'y0'
    nsteps = 1000
    xarr = np.linspace(0.0, 1.0, nsteps)
    # Stimulus tables allow you to store sequences of numbers which
    # are delivered via the 'output' message at each time step. This
    # is a placeholder and in real scenario you will be using any
    # sourceFinfo that sends out a double value.
    input_x = moose.StimulusTable('/xtab')
    input_x.vector = xarr
    input_x.startTime = 0.0
    input_x.stepPosition = xarr[0]
    input_x.stopTime = simtime
    moose.connect(input_x, 'output', function.x[0], 'input')

    yarr = np.linspace(-np.pi, np.pi, nsteps)
    input_y = moose.StimulusTable('/ytab')
    input_y.vector = yarr
    input_y.startTime = 0.0
    input_y.stepPosition = yarr[0]
    input_y.stopTime = simtime
    moose.connect(function, 'requestOut', input_y, 'getOutputValue')

    # data recording
    result = moose.Table('/ztab')
    moose.connect(result, 'requestOut', function, 'getValue')
    derivative = moose.Table('/zprime')
    moose.connect(derivative, 'requestOut', function, 'getDerivative')
    rate = moose.Table('/dz_by_dt')
    moose.connect(rate, 'requestOut', function, 'getRate')
    x_rec = moose.Table('/xrec')
    moose.connect(x_rec, 'requestOut', input_x, 'getOutputValue')
    y_rec = moose.Table('/yrec')
    moose.connect(y_rec, 'requestOut', input_y, 'getOutputValue')

    dt =  simtime/nsteps
    for ii in range(32):
        moose.setClock(ii, dt)
    moose.reinit()
    moose.start(simtime)

    # Uncomment the following lines and the import matplotlib.pyplot as plt on top
    # of this file to display the plot.
    plt.subplot(3,1,1)
    plt.plot(x_rec.vector, result.vector, 'r-', label='z = {}'.format(function.expr))
    z = function.c['c0'] * np.exp(function.c['c1'] * xarr) * np.cos(yarr) + np.sin(np.arange(len(xarr)) * dt)
    plt.plot(xarr, z, 'b--', label='numpy computed')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.legend()

    plt.subplot(3,1,2)
    plt.plot(y_rec.vector, derivative.vector, 'r-', label='dz/dy0')
    # derivatives computed by putting x values in the analytical formula
    dzdy = function.c['c0'] * np.exp(function.c['c1'] * xarr) * (- np.sin(yarr))
    plt.plot(yarr, dzdy, 'b--', label='numpy computed')
    plt.xlabel('y')
    plt.ylabel('dz/dy')
    plt.legend()

    plt.subplot(3,1,3)
    # *** BEWARE *** The first two entries are spurious. Entry 0 is
    # *** from reinit sending out the defaults. Entry 2 is because
    # *** there is no lastValue for computing real forward difference.
    plt.plot(np.arange(2, len(rate.vector), 1) * dt, rate.vector[2:], 'r-', label='dz/dt')
    dzdt = np.diff(z)/dt
    plt.plot(np.arange(0, len(dzdt), 1.0) * dt, dzdt, 'b--', label='numpy computed')
    plt.xlabel('t')
    plt.ylabel('dz/dt')
    plt.legend()
    plt.tight_layout()
    plt.savefig(__file__+'.png')

if __name__ == '__main__':
    test_example()
