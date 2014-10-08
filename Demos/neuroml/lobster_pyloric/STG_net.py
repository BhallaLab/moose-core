## Aditya Gilra, NCBS, Bangalore, 2013

"""
Inside the .../Demos/neuroml/lobster_ploric/ directory supplied with MOOSE, run
python STG_net.py
(other channels and morph xml files are already present in this same directory).
The soma name below is hard coded for gran98, else any other file can be used by modifying this script.
"""

#import os
#os.environ['NUMPTHREADS'] = '1'
import sys
sys.path.extend(['../../../python','synapses'])

import moose
from moose.utils import *
from moose.neuroml.NeuroML import NeuroML

from pylab import *

simdt = 25e-6 # s
plotdt = 25e-6 # s
runtime = 10.0 # s
cells_path = '/cells' # neuromlR.readNeuroMLFromFile creates cells in '/cells'

# for graded synapses, else NeuroML event-based are used
from load_synapses import load_synapses
moose.Neutral('/library')
# set graded to False to use event based synapses
#  if False, neuroml event-based synapses get searched for and loaded
# True to load graded synapses
graded_syn = True
#graded_syn = False
if graded_syn:
    load_synapses()

def loadSTGNeuroML_L123(filename):
    neuromlR = NeuroML()
    ## readNeuroMLFromFile below returns:
    # This returns
    # populationDict = {
    #     'populationname1':('cellName',{('instanceid1'):moosecell, ... }) 
    #     , ... 
    #     }
    # (cellName and instanceid are strings, mooosecell is a moose.Neuron object instance)
    # and
    # projectionDict = { 
    #     'projName1':('source','target',[('syn_name1','pre_seg_path','post_seg_path')
    #     ,...]) 
    #     , ... 
    #     }
    populationDict, projectionDict = \
        neuromlR.readNeuroMLFromFile(filename)
    soma1_path = populationDict['AB_PD'][1][0].path+'/Soma_0'
    soma1Vm = setupTable('somaVm',moose.Compartment(soma1_path),'Vm')
    soma2_path = populationDict['LP'][1][0].path+'/Soma_0'
    soma2Vm = setupTable('somaVm',moose.Compartment(soma2_path),'Vm')
    soma3_path = populationDict['PY'][1][0].path+'/Soma_0'
    soma3Vm = setupTable('somaVm',moose.Compartment(soma3_path),'Vm')

    # monitor channel current
    channel_path = soma1_path + '/KCa_STG'
    channel_Ik = setupTable('KCa_Ik',moose.element(channel_path),'Ik')
    # monitor Ca
    capool_path = soma1_path + '/CaPool_STG'
    capool_Ca = setupTable('CaPool_Ca',moose.element(capool_path),'Ca')

    # monitor synaptic current
    soma2 = moose.element(soma2_path)
    print "Children of",soma2_path,"are:"
    for child in soma2.children:
        print child.className, child.path
    if graded_syn:
        syn_path = soma2_path+'/DoubExpSyn_Ach__cells-0-_AB_PD_0-0-_Soma_0'
        syn = moose.element(syn_path)
    else:
        syn_path = soma2_path+'/DoubExpSyn_Ach'
        syn = moose.element(syn_path)
    syn_Ik = setupTable('DoubExpSyn_Ach_Ik',syn,'Ik')

    print "Reinit MOOSE ... "
    resetSim(['/elec',cells_path], simdt, plotdt, simmethod='ee')

    print "Running ... "
    moose.start(runtime)
    tvec = arange(0.0,runtime+2*plotdt,plotdt)
    tvec = tvec[ : soma1Vm.vector.size ]
    
    figure(facecolor='w')
    plot(tvec,soma1Vm.vector,label='AB_PD',color='g',linestyle='solid')
    plot(tvec,soma2Vm.vector,label='LP',color='r',linestyle='solid')
    plot(tvec,soma3Vm.vector,label='PY',color='b',linestyle='solid')
    legend()
    title('Soma Vm')
    xlabel('time (s)')
    ylabel('Voltage (V)')

    figure(facecolor='w')
    plot(tvec,channel_Ik.vector,color='b',linestyle='solid')
    title('KCa current; Ca conc')
    xlabel('time (s)')
    ylabel('Ik (Amp)')
    twinx()
    plot(tvec,capool_Ca.vector,color='r',linestyle='solid')
    ylabel('Ca (mol/m^3)')

    figure(facecolor='w')
    plot(tvec,syn_Ik.vector,color='b',linestyle='solid')    
    title('Ach syn current in '+soma2_path)
    xlabel('time (s)')
    ylabel('Isyn (S)')
    print "Showing plots ..."
    show()

filename = "Generated.net.xml"
if __name__ == "__main__":
    if len(sys.argv)>=2:
        filename = sys.argv[1]
    loadSTGNeuroML_L123(filename)
