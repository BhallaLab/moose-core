# test_hsolve_tcr.py --- 
# 
# Filename: test_hsolve_tcr.py
# Description: 
# Author: 
# Maintainer: 
# Created: Wed Jun 12 11:10:44 2013 (+0530)
# Version: 
# Last-Updated: Sat Jun 22 16:02:37 2013 (+0530)
#           By: subha
#     Update #: 179
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
import sys
sys.path.append('/home/subha/src/moose/python')
import os
from datetime import datetime
import unittest
import numpy as np
import pylab

import moose
import config

from cell_test_util import setup_current_step_model, SingleCellCurrentStepTest
import testutils
import cells
from moose import utils

simdt = 1e-6
plotdt = 0.25e-3
simtime = 1000e-3
    

# pulsearray = [[1.0, 100e-3, 1e-9],
#               [0.5, 100e-3, 0.3e-9],
#               [0.5, 100e-3, 0.1e-9],
#               [0.5, 100e-3, -0.1e-9],
#               [0.5, 100e-3, -0.3e-9]]

pulsearray = [[500e-3, 20e-3, 1e-9]]


class TestHSolveTCR(unittest.TestCase):
    def setUp(self):
        self.model_container = moose.Neutral('/model')
        self.data_container = moose.Neutral('/data')
        self.m1 = moose.Neutral('%s/hsolve_tcr' % (self.model_container.path))
        self.d1 = moose.Neutral('%s/hsolve_tcr' % (self.data_container.path))
        self.m2 = moose.Neutral('%s/ee_tcr' % (self.model_container.path))
        self.d2 = moose.Neutral('%s/ee_tcr' % (self.data_container.path))
        self.p1 = setup_current_step_model(self.m1, self.d1, 'TCR', pulsearray, simdt, plotdt)
        self.p2 = setup_current_step_model(self.m2, self.d2, 'TCR', pulsearray, simdt, plotdt)
        for ii in moose.wildcardFind('%s/##[ISA=ChanBase]' % (self.model_container.path)):
            moose.delete(ii)
        for ii in moose.wildcardFind('%s/##[ISA=CaConc]' % (self.model_container.path)):
            moose.delete(ii)
        for ii in moose.wildcardFind('%s/##[ISA=Compartment]' % (self.model_container.path)):
            moose.element(ii).Ra = 1e6
            
        self.hsolve = moose.HSolve('%s/solve' % (self.p1['cell'].path))
        self.hsolve.dt = simdt
        self.hsolve.target = self.p1['cell'].path
        
    def testVmSeriesPlot(self):
        utils.setDefaultDt(elecdt=simdt)
        utils.assignDefaultTicks(modelRoot=self.m2.path)
        utils.assignTicks({0:'%s/##[ISA=HSolve]' % (self.m1.path),
                           1: '%s/##[ISA=PulseGen]' % (self.m1.path),
                           7: '%s/##[ISA=VClamp]' % (self.m1.path)})
        moose.reinit()
        tick = moose.element('/clock/tick')
        for ii in range(10):
            field = 'proc%d' % (ii)
            print 'Connected to', field
            for n in tick.neighbours[field]:
                print '\t', n
        ts = datetime.now()
        utils.stepRun(simtime, 10000*simdt, verbose=True, logger=config.logger)
        te = datetime.now()
        td = te - ts
        t1 = np.linspace(0, simtime, len(self.p1['somaVm'].vec))
        t2 = np.linspace(0, simtime, len(self.p2['somaVm'].vec))
        config.logger.info('Finished simulation of %g s with %d process threads and dt=%d s in %g s' % (simtime, moose.NUMPTHREADS, simdt, td.seconds + 1e-6 * td.microseconds))
        hsolve_data_file = os.path.join(config.data_dir, 'hsolve_tcr_%s.csv' % (config.filename_suffix))
        eeuler_data_file = os.path.join(config.data_dir, 'ee_tcr_%s.csv' % (config.filename_suffix))        
        np.savetxt(hsolve_data_file, np.vstack((t1, self.p1['somaVm'].vec)).transpose())
        np.savetxt(eeuler_data_file, np.vstack((t2, self.p2['somaVm'].vec)).transpose())
        config.logger.info('Saved Exp Euler data in %s\nHSolve data in %s' % (hsolve_data_file, eeuler_data_file))
        pylab.plot(t1, self.p1['somaVm'].vec, label='hsolve')
        pylab.plot(t2, self.p2['somaVm'].vec, label='euler')
        pylab.legend()
        pylab.show()
                           
        


if __name__ == '__main__':
    unittest.main()




# 
# test_hsolve_tcr.py ends here
