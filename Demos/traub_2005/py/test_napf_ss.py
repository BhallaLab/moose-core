# test_napf_ss.py --- 
# 
# Filename: test_napf_ss.py
# Description: 
# Author: 
# Maintainer: 
# Created: Fri Jun  1 16:27:07 2012 (+0530)
# Version: 
# Last-Updated: Fri Jun  1 16:30:49 2012 (+0530)
#           By: subha
#     Update #: 4
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

import os
os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.append('../../../python')
import numpy as np

from testutils import *
import nachans


class TestNaPF_SS(ChannelTestBase):
    channelname = 'NaPF_SS'
    params = run_single_channel(channelname, 1e-9, simtime)
    vm = np.asarray(params['Vm'].vec)        
    gk = np.asarray(params['Gk'].vec)
    tseries = np.array(range(0, len(params['Vm'].vec))) * simdt
    def testNaPF_SS_Vm_Moose(self):
        print 'Testing MOOSE Vm  ...',
        err = compare_channel_data(self.vm, self.channelname, 'Vm', 'moose', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'
        
    def testNaPF_SS_Gk_Moose(self):
        print 'Testing MOOSE Gk  ...',
        err = compare_channel_data(self.gk, self.channelname, 'Gk', 'moose', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.05)
        print 'OK'

    def testNaPF_SS_Vm_Neuron(self):
        print 'Testing NEURON Vm  ...',
        data = np.c_[self.tseries, self.vm]
        err = compare_channel_data(data, self.channelname, 'Vm', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)        
        print 'OK'

    def testNaPF_SS_Gk_Neuron(self):
        print 'Testing NEURON Gk  ...',
        data = np.c_[self.tseries, self.gk]
        err = compare_channel_data(data, self.channelname, 'Gk', 'neuron', x_range=(simtime/10.0, simtime), plot=True)
        self.assertLess(err, 0.05)
        print 'OK'


if __name__ == '__main__':
    unittest.main()

# 
# test_napf_ss.py ends here
