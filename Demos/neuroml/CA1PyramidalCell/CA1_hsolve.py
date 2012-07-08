## Aditya Gilra, NCBS, Bangalore, 2012

"""
Inside the .../Demos/CA1PyramidalCell/ directory supplied with MOOSE, run
python testNeuroML_CA1.py
(other channels and morph xml files are already present in this same directory).
The soma name below is hard coded for CA1, else any other file can be used by modifying this script.
"""

import moose
from moose.utils import *

from moose.neuroml.NeuroML import NeuroML

from pylab import *

simdt = 10e-6 # s
plotdt = 10e-6 # s
runtime = 0.2 # s
cells_path = '/cells' # neuromlR.readNeuroMLFromFile creates cells in '/cells'

def loadGran98NeuroML_L123(filename):
    neuromlR = NeuroML()
    populationDict, projectionDict = \
        neuromlR.readNeuroMLFromFile(filename)
    soma_path = populationDict['CA1group'][1][0].path+'/Seg0_soma_0_0'
    somaVm = setupTable('somaVm',moose.Compartment(soma_path),'Vm')
    #somaCa = setupTable('somaCa',moose.CaConc(soma_path+'/Gran_CaPool_98'),'Ca')
    #somaIKCa = setupTable('somaIKCa',moose.HHChannel(soma_path+'/Gran_KCa_98'),'Gk')
    #KDrX = setupTable('ChanX',moose.HHChannel(soma_path+'/Gran_KDr_98'),'X')
    soma = moose.Compartment(soma_path)
    
    h = moose.HSolve( cells_path+'/solve' )
    h.dt = simdt
    h.path = cells_path

    print "Reinit MOOSE ... "
    resetSim(['/elec','/cells'],simdt,plotdt,hsolve_path=cells_path+'/solve') # from moose.utils
    print "Running ... "
    moose.start(runtime)
    tvec = arange(0.0,runtime,simdt)
    plot(tvec,somaVm.vec[1:])
    title('Soma Vm')
    xlabel('time (s)')
    ylabel('Voltage (V)')
    print "Showing plots ..."
    show()

if __name__ == "__main__":
    if len(sys.argv)<2:
        filename = "CA1.net.xml"
    else:
        filename = sys.argv[1]
loadGran98NeuroML_L123(filename)
