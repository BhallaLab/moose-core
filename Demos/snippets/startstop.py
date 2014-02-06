# startstop.py --- 
# 
# Filename: startstop.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sun Jul  1 13:42:28 2012 (+0530)
# Version: 
# Last-Updated: Sun Jul  1 14:41:34 2012 (+0530)
#           By: subha
#     Update #: 62
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

import time
import pylab

import sys
sys.path.append('../../python')
import moose

def setup_model():
    model_container = moose.Neutral('/model')    
    pulse = moose.PulseGen('/model/pulse')
    pulse.level[0] = 1.0
    pulse.delay[0] = 0.5
    pulse.width[0] = 0.5
    table = moose.Table('%s/tab' % (pulse.path))
    moose.connect(table, 'requestOut', pulse, 'getOutput')
    moose.setClock(0, 0.1)
    moose.setClock(1, 0.1)
    moose.setClock(2, 0.1)
    moose.useClock(0, '%s,%s' % (pulse.path, table.path), 'process')
    return table
            
def startstop_demo():
    """Starts a simulation for some steps, pauses midway with the
    current data, runs till end."""
    table = setup_model()
    moose.reinit()
    clock = moose.element('/clock')
    dt = clock.tick[0].dt
    print dt
    steps = 100
    simtime = dt * steps
    print 'Total simulation time', simtime
    t = simtime/2.0
    print 'Running simulation for', t, 'seconds'
    moose.start(t)
    # moose.stop() # Somehow this does not work in single threaded Python script
    pylab.plot(pylab.linspace(0, clock.currentTime, len(table.vector)), table.vector)    
    pylab.show()
    print 'Simulator time:', clock.currentTime
    # How much of the total simulation time is left?
    time_left_to_run = simtime - clock.currentTime
    moose.start(time_left_to_run)
    print 'Simulator time at end of simulation', clock.currentTime
    pylab.plot(pylab.linspace(0, clock.currentTime, len(table.vector)), table.vector)    
    pylab.show()

if __name__ == '__main__':
    startstop_demo()

# 
# startstop.py ends here
