# test_km.py --- 
# 
# Filename: test_km.py
# Description: 
# Author: 
# Maintainer: 
# Created: Thu May 31 11:34:59 2012 (+0530)
# Version: 
# Last-Updated: Thu May 31 13:45:34 2012 (+0530)
#           By: subha
#     Update #: 10
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
from testutils import *
from kchans import *

class TestKM(ChannelTestBase):
    channelname = 'KM'
    params = run_single_channel(channelname, 1e-9, simtime)
    vm = np.array(params['Vm'].vec)
    gk = np.array(params['Gk'].vec)
    tseries = np.arange(0, len(vm), 1.0) * simdt
    
    def testKM_Vm_Neuron(self):
        print 'Testing NEURON Vm ...',
        data = np.c_[self.tseries, self.vm]
        err = compare_channel_data(data, TestKM.channelname, 'Vm', 'neuron', x_range=(simtime/10.0, simtime))
        self.assertLess(err, 0.01)
        print 'OK'

    def testKM_Gk_Neuron(self):
        print 'Testing NEURON Gk ...',
        data = np.c_[self.tseries, self.gk]
        err = compare_channel_data(data, TestKM.channelname, 'Gk', 'neuron', x_range=(simtime/10.0, simtime), plot=True)
        self.assertLess(err, 0.01)
        print 'OK'

if __name__ == '__main__':
    unittest.main()


# 
# test_km.py ends here
