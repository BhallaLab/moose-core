## Aditya Gilra, NCBS, Bangalore, 2013

"""
Inside the .../Demos/neuroml/lobster_ploric/ directory supplied with MOOSE, run
python STG_net.py
(other channels and morph xml files are already present in this same directory).
The soma name below is hard coded for gran98, else any other file can be used by modifying this script.
"""

import os
os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.append('../../../python')

import moose
from moose.utils import *
from moose.neuroml.NeuroML import NeuroML

from pylab import *

simdt = 25e-6 # s
plotdt = 25e-6 # s
runtime = 10.0 # s
cells_path = '/cells' # neuromlR.readNeuroMLFromFile creates cells in '/cells'

def loadSTGNeuroML_L123(filename):
    neuromlR = NeuroML()
    populationDict, projectionDict = \
        neuromlR.readNeuroMLFromFile(filename)
    soma1_path = populationDict['AB_PD'][1][0].path+'/Soma_0'
    soma1Vm = setupTable('somaVm',moose.Compartment(soma1_path),'Vm')
    soma2_path = populationDict['LP'][1][0].path+'/Soma_0'
    soma2Vm = setupTable('somaVm',moose.Compartment(soma2_path),'Vm')
    soma3_path = populationDict['PY'][1][0].path+'/Soma_0'
    soma3Vm = setupTable('somaVm',moose.Compartment(soma3_path),'Vm')

    print "Reinit MOOSE ... "
    resetSim(['/elec',cells_path], simdt, plotdt, simmethod='hsolve')

    print "Running ... "
    moose.start(runtime)
    tvec = arange(0.0,runtime+2*plotdt,plotdt)
    tvec = tvec[ : soma1Vm.vec.size ]
    plot(tvec,soma1Vm.vec,label='AB_PD',color='g',linestyle='dashed')
    plot(tvec,soma2Vm.vec,label='LP',color='r',linestyle='solid')
    plot(tvec,soma3Vm.vec,label='PY',color='b',linestyle='dashed')
    legend()
    title('Soma Vm')
    xlabel('time (s)')
    ylabel('Voltage (V)')
    print "Showing plots ..."
    show()

filename = "Generated.net.xml"
if __name__ == "__main__":
    if len(sys.argv)>=2:
        filename = sys.argv[1]
loadSTGNeuroML_L123(filename)
