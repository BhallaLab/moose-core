# test_deepaxoaxonic.py --- 
# 
# Filename: test_deepaxoaxonic.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Jul 16 16:12:55 2012 (+0530)
# Version: 
# Last-Updated: Thu Nov  8 17:51:49 2012 (+0530)
#           By: subha
#     Update #: 494
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

# Code:

import unittest
from cell_test_util import SingleCellCurrentStepTest
import testutils
import cells
from moose import utils

simdt = 5e-6
plotdt = 0.25e-3
simtime = 1.0
    

# pulsearray = [[1.0, 100e-3, 1e-9],
#               [0.5, 100e-3, 0.3e-9],
#               [0.5, 100e-3, 0.1e-9],
#               [0.5, 100e-3, -0.1e-9],
#               [0.5, 100e-3, -0.3e-9]]



class TestDeepAxoaxonic(SingleCellCurrentStepTest):
    def __init__(self, *args, **kwargs):
        self.celltype = 'DeepAxoaxonic'
        SingleCellCurrentStepTest.__init__(self, *args, **kwargs)
        self.pulse_array = [(100e-3, 100e-3, 1e-9),
                            (1e9, 0, 0)]
        # self.solver = 'ee'
        self.simdt = simdt
        self.plotdt = plotdt

    def setUp(self):
        SingleCellCurrentStepTest.setUp(self)

    def testVmSeriesPlot(self):
        self.runsim(simtime, pulsearray=self.pulse_array)
        self.plot_vm()

    def testChannelDensities(self):
        pass
        # equal = compare_cell_dump(self.dump_file, '../nrn/'+self.dump_file)
        # self.assertTrue(equal)


if __name__ == '__main__':
    unittest.main()



# 
# test_deepaxoaxonic.py ends here
