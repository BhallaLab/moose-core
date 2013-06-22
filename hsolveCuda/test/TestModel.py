'''
Created on Apr 30, 2013

@author: saeed
'''

import sys
sys.path.append('../../python')
sys.path.append('../cudaLibrary/')

from ctypes import *
lib = CDLL('../cudaLibrary/libmooseCudaLibrary.so')
#import pdb
#pdb.set_trace()
import moose

if __name__ == '__main__':
    # Create Model
    model = moose.Neutral('/model')
    soma = moose.Compartment('/model/soma')
    pulse = moose.PulseGen('/model/pulse')
    data = moose.Neutral('/data')
    vmtab = moose.Table('/data/soma_Vm')
    
    soma.Cm = 1e-9
    soma.Rm = 1e7
    soma.initVm = -0.07
    
    pulse.delay[0] = 50e-3
    pulse.width[0] = 100e-3
    pulse.level[0] = 1e-9
    pulse.delay[1] = 1e9
    
    moose.connect(pulse, 'outputOut', soma, 'injectMsg')
    moose.connect(vmtab, 'requestData', soma, 'get_Vm')
    
    moose.setClock(0, 0.025e-3)
    moose.setClock(1, 0.025e-3)
    moose.setClock(2, 0.25e-3)
    
    moose.useClock(2, '/data/soma_Vm', 'process')
    moose.useClock(0, soma.path, 'init')
    moose.useClock(1, '/model/##', 'process')
    
    moose.reinit()
#     moose.start(1000e-3)
#       
#     import pylab
#     t = pylab.linspace(0, 300e-3, len(vmtab.vec))
#     pylab.plot(t, vmtab.vec)
#     pylab.show()
    
    