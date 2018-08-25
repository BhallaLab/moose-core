# -*- coding: utf-8 -*-
# test_function.py ---
#
# Filename: test_function.py
# Description:
# Author: subha
# Maintainer:
# Created: Sat Mar 28 19:34:20 2015 (-0400)
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
"""Check variable ordering - bug #161"""
from __future__ import print_function

import numpy as np
import moose
print( "[INFO ] Using moose %s form %s" % (moose.version(), moose.__file__) )

def test_var_order():
    """The y values are one step behind the x values because of
    scheduling sequences"""
    nsteps = 5
    simtime = nsteps
    dt = 1.0
    # fn0 = moose.Function('/fn0')
    # fn0.x.num = 2
    # fn0.expr = 'x0 + x1'
    # fn0.mode = 1
    fn1 = moose.Function('/fn1')
    fn1.x.num = 2
    fn1.expr = 'y1 + y0 + x1 + x0'
    fn1.mode = 1
    inputs = np.arange(0, nsteps+1, 1.0)
    x0 = moose.StimulusTable('/x0')
    x0.vector = inputs
    x0.startTime = 0.0
    x0.stopTime = simtime
    x0.stepPosition = 0.0
    inputs /= 10
    x1 = moose.StimulusTable('/x1')
    x1.vector = inputs
    x1.startTime = 0.0
    x1.stopTime = simtime
    x1.stepPosition = 0.0
    inputs /= 10
    y0 = moose.StimulusTable('/y0')
    y0.vector = inputs
    y0.startTime = 0.0
    y0.stopTime = simtime
    y0.stepPosition = 0.0
    inputs /= 10
    y1 = moose.StimulusTable('/y1')
    y1.vector = inputs
    y1.startTime = 0.0
    y1.startTime = 0.0
    y1.stopTime = simtime
    y1.stepPosition = 0.0
    moose.connect(x0, 'output', fn1.x[0], 'input')
    moose.connect(x1, 'output', fn1.x[1], 'input')
    moose.connect(fn1, 'requestOut', y0, 'getOutputValue')
    moose.connect(fn1, 'requestOut', y1, 'getOutputValue')
    z1 = moose.Table('/z1')
    moose.connect(z1, 'requestOut', fn1, 'getValue')
    for ii in range(32):
        moose.setClock(ii, dt)
    moose.reinit()
    moose.start(simtime)
<<<<<<< HEAD
    for ii in range(len(z1.vector)):
        print(ii, z1.vector[ii])
=======
    expected = [0, 1.1, 2.211, 3.322, 4.433, 5.544]
    assert np.allclose(z1.vector, expected), "Excepted %s, got %s" % (expected, z1.vector )

def test_t( ):
    moose.delete( '/' )
    f = moose.Function( '/funct' )
    f.expr = 't'
    t = moose.Table( '/table1' )
    moose.setClock( f.tick, 0.1)
    moose.setClock( t.tick, 0.1)
    moose.connect( t, 'requestOut', f, 'getValue' )
    moose.reinit()
    moose.start( 1 )
    y = t.vector
    print( 'Raw', y )
    print( 'Diff', np.diff(y) )
    assert (np.diff(y) == 0.1).all(), np.diff(y)
>>>>>>> c5fded807... removed npy support. On large files, it was not working properly. Only



if __name__ == '__main__':
    test_t( )
    test_var_order()
