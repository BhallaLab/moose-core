# function.py --- 
# 
# Filename: function.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Tue Sep  9 17:59:50 2014 (+0530)
# Version: 
# Last-Updated: 
#           By: 
#     Update #: 0
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 
# 
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 
# 

# Code:

import numpy as np
import sys
# import matplotlib.pyplot as plt

import moose

simtime = 1.0

def example():
    """Function objects can be used to evaluate expressions with arbitrary
    number of variables and constants. We can assign expression of the
    form::

        f(c0, c1, ..., cM, x0, x1, ..., xN, y0,..., yP ) 

    where `ci`'s are constants and `xi`'s and `yi`'s are variables.

    The constants must be defined before setting the expression and
    variables are connected via messages. The constants can have any
    name, but the variable names must be of the form x{i} or y{i}
    where i is increasing integer starting from 0. 

    The `xi`'s are field elements and you have to set their number
    first (function.x.num = N). Then you can connect any source field
    sending out double to the 'setVar' destination field of the
    `x[i]`.

    The `yi`'s are useful when the required variable is a value field
    and is not available as a source field. In that case you connect
    the `requestOut` source field of the function element to the
    `get{Field}` destination field on the target element. The `yi`'s
    are automatically added on connecting. Thus, if you call::

       moose.connect(function, 'requestOut', a, 'getSomeField')
       moose.connect(function, 'requestOut', b, 'getSomeField')

    then ``a.someField`` will be assigned to ``y0`` and
    ``b.someField`` will be assigned to ``y1``.

    In this example we evaluate the expression: ``z = c0 * exp(c1 *
    x0) * cos(y0)``

    with x0 ranging from -1 to +1 and y0 ranging from -pi to
    +pi. These values are stored in two stimulus tables called xtab
    and ytab respectively, so that at each timestep the next values of
    x0 and y0 are assigned to the function.

    Unlike Func class, the number of variables and constants are
    unlimited in Function and you can set all the variables via
    messages.

    """
    function = moose.Function('/function')
    function.c['c0'] = 1.0
    function.c['c1'] = 2.0
    function.x.num = 2
    function.expr = 'c0 * exp(c1*x0) * cos(y0)'
    # mode 1 - just evaluate function value,
    # mode 2 - evaluate derivative,
    # mode 3 evaluate both
    function.mode = 3 
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
    moose.connect(input_x, 'output', function.x[0], 'setVar')
    yarr = np.linspace(-np.pi, np.pi, nsteps)
    input_y = moose.StimulusTable('/ytab')
    input_y.vector = yarr
    input_y.startTime = 0.0
    input_y.stepPosition = yarr[0]
    input_y.stopTime = simtime
    moose.connect(function, 'requestOut', input_y, 'getOutputValue')
    result = moose.Table('/ztab')
    moose.connect(result, 'requestOut', function, 'getValue')
    derivative = moose.Table('/zprime')
    moose.connect(derivative, 'requestOut', function, 'getDerivative')
    dt =  simtime/nsteps
    moose.setClock(0, dt)
    moose.useClock(0, '/xtab,/ytab,/ztab,/zprime,/function', 'process')
    moose.reinit()    
    moose.start(simtime)
    
    # Uncomment the following lines and the import matplotlib.pyplot as plt on top
    # of this file to display the plot.
    
    # plt.plot(result.vector, 'b-', label='z = c0 * exp(c1 * x0) * cos(y0)')
    # plt.plot(derivative.vector, 'g-', label='dz/dx0')
    # z = function.c['c0'] * np.exp(function.c['c1'] * xarr) * np.cos(yarr)
    # plt.plot(z, 'r--', label='numpy computed')
    # plt.legend()
    # plt.show()
    
if __name__ == '__main__':
    example()
    
    

# 
# function.py ends here
