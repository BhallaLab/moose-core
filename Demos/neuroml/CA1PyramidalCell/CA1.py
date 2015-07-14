## Aditya Gilra, NCBS, Bangalore, 2012

"""
Inside the .../Demos/CA1PyramidalCell/ directory supplied with MOOSE, run
python CA1.py <filename>
(if no filename is specified, the single compartment CA1 cell is used.)
(supporting channels and different morph xml files are already present in this same directory).
The soma name below is hard coded for CA1, else any other file can be used by modifying this script.
"""

import os
os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.append('../../../python/')
import moose
from moose.utils import *
import moose.utils as mu

from moose.neuroml.NeuroML import NeuroML

from pylab import *

simdt = 10e-6 # s
plotdt = 10e-6 # s
runtime = 0.2 # s

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
    print("Reinit MOOSE ... ")
    resetSim(['/elec','/cells'],simdt,plotdt,simmethod='ee') # from moose.utils
    print("Running ... ")
    moose.start(runtime)
    tvec = arange(0.0,runtime,simdt)
    plot(tvec, somaVm.vector[1:])
    res =  mu.spike_train_simple_stat( somaVm.vector )
    if res['number of spikes'] != 9:
        print(("[WARN] Expecting 9 spikes, got %s" % res['number of spikes']))
    title('Soma Vm')
    xlabel('time (s)')
    ylabel('Voltage (V)')
    print("Showing plots ...")
    show( block = False )
    outfile = "%s.png" % __file__
    print("Writing results to %s" % outfile)
    savefig(outfile)

if __name__ == "__main__":
    if len(sys.argv)<2:
        filename = "CA1soma.net.xml"
    else:
        filename = sys.argv[1]
    loadGran98NeuroML_L123(filename)
