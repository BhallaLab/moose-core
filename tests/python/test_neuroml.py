"""ca1_test.py: 

    Testing scripts for CA1.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"


import unittest
import platform

from neuroml.FvsI_CA1 import ca1_main, loadModel
from neuroml.CA1 import loadGran98NeuroML_L123


class CA1Test( unittest.TestCase ):

    def test_injected_current_and_frequency(self):
        loadModel('./neuroml/cells_channels/CA1soma.morph.xml')
        self.assertEqual( 10, ca1_main(4.0e-13))
        self.assertEqual( 20, ca1_main(8.0e-13))
        self.assertEqual( 29, ca1_main(14.0e-13))
        self.assertEqual( 34, ca1_main(18.0e-13))

    def test_injected_current_and_spikes(self):
        #self.assertEqual( 00, ca1_main('./neuroml/cells_channels/CA1.morph.xml', 2.0e-13))
        self.assertEqual(9, loadGran98NeuroML_L123('neuroml/CA1soma.net.xml'))


    
if __name__ == '__main__':
    unittest.main()
