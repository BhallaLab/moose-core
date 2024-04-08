# test_granule98.py --- 
# 
# Filename: test_granule98.py
# Description: 
# Author: Subhasis Ray
# Created: Mon Apr  8 21:41:22 2024 (+0530)
# Last-Updated: Mon Apr  8 22:00:04 2024 (+0530)
#           By: Subhasis Ray
# 

# Code:
"""Test code for the Granule cell model

"""
import os
import unittest
import logging


LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO')
logging.basicConfig(level=LOGLEVEL)


import moose
from moose.neuroml2.reader import NML2Reader


class TestGran98(unittest.TestCase):
    def setUp(self):
        self.reader = NML2Reader()
        self.lib = moose.Neutral('/library')
        self.filename = 'test_files/Granule_98/GranuleCell.net.nml'
        self.reader.read(self.filename)
        # for ncell in self.reader.nml_to_moose:
        #     if isinstance(ncell, nml.Cell):
        #         self.ncell = ncell
        #         break
        self.mcell = moose.element(moose.wildcardFind('/##[ISA=Neuron]')[0])
        moose.le(self.mcell.path)


    def test_CaPool(self):
        pass

    
if __name__ == '__main__':
    unittest.main()




# 
# test_granule98.py ends here
