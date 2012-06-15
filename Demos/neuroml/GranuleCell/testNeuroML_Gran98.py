## Aditya Gilra, NCBS, Bangalore, 2012

"""
Inside the .../Demos/GranuleCell/ directory supplied with MOOSE, run
python testNeuroML_Gran98.py
(other channels and morph xml files are already present in this same directory).
The soma name below is hard coded for gran98, else any other file can be used by modifying this script.
"""

import moose
from moose.utils import *
from moose.neuroml.NeuroML import NeuroML

from pylab import *

def loadGran98NeuroML_L123(filename):
    neuromlR = NeuroML()
    populationDict, projectionDict = \
        neuromlR.readNeuroMLFromFile(filename)
    soma_path = populationDict['Gran'][1][0].path+'/Soma_0'
    somaVm = setupTable('somaVm',moose.Compartment(soma_path),'VmOut')
    resetSim(['/elec','/cells'],50e-6,50e-6) # from moose.utils
    moose.start(1.0)
    plot(somaVm.vec)
    print "Showing",soma_path,"Vm"
    show()

if __name__ == "__main__":
    if len(sys.argv)<2:
        filename = "Generated.net.xml"
    else:
        filename = sys.argv[1]
    loadGran98NeuroML_L123(filename)
