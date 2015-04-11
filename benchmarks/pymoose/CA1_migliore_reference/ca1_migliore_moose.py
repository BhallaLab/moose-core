"""ca1_migliore.py: 

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import moose
import moose.neuroml as nml
import pylab

def main():
    modelName = "./Generated.net.xml"
    nml.loadNeuroML_L123(modelName)

if __name__ == '__main__':
    main()
