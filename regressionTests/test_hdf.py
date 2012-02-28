# test_hdf.py --- 
# 
# Filename: test_hdf.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Mon Feb 27 19:26:56 2012 (+0530)
# Version: 
# Last-Updated: Wed Feb 29 00:47:46 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 59
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

# Code:

import sys
import os
import unittest
import h5py

import moose

class HDF5Demo(unittest.TestCase):
    def setUp(self):
        filepath = 'test_hdf.h5'
        comp_path = '/comp_hdf5'
        table_path = '/data/tab_hdf5'
        hdf_path = '/data/hdf5'
        try:
            if os.access(filepath, os.W_OK):
                os.remove(filepath)
        except Exception:
            pass
                
        self.comp = moose.Compartment(comp_path)
        self.comp.inject = 0.1
        self.data_container = moose.Neutral('/data')
        self.table = moose.Table(table_path)
        moose.connect(self.table, 'requestData', self.comp, 'get_Vm')
        self.hdf = moose.HDF5DataWriter(hdf_path)
        # self.hdf.filename = filepath
        moose.connect(self.hdf, 'requestData', self.table, 'get_vec')
        # moose.connect(self.hdf, 'clear', self.table, 'clearVec')
        moose.setClock(0, 1.0)
        moose.setClock(1, 1.0)
        moose.setClock(2, 3.0)
        moose.useClock(0, comp_path, 'init')
        moose.useClock(1, '%s,%s' % (comp_path, table_path),'process')
        moose.useClock(2, '%s' % (hdf_path), 'process')
        moose.reinit()
        moose.start(8.0)
        self.hdf.flush()
    # def testMode(self):
    #     self.assertEqual(self.hdf.mode, 4)

    def testDataset(self):
        pass
        # h5 = h5py.File(self.hdf.filename, 'r')
        # data = h5['/data/tab_hdf5']
        # self.assertEqual(len(data), 6)
        # h5.close()

if __name__ == '__main__':
    unittest.main()

# 
# test_hdf.py ends here
