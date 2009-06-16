# brutespinstell.py --- 
# 
# Filename: brutespinstell.py
# Description: This is an unelegant version of spiny stellate cells
# Author: subhasis ray
# Maintainer: 
# Created: Fri May  8 11:24:30 2009 (+0530)
# Version: 
# Last-Updated: Mon Jun  1 16:35:50 2009 (+0530)
#           By: subhasis ray
#     Update #: 196
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
# 
 

# Code:

from collections import deque, defaultdict
from datetime import datetime
import moose

from kchans import *
from nachans import *
from cachans import *
from capool import *
from archan import *

from compartment import MyCompartment

class SpinyStellate(moose.Cell):
    ENa = 50e-3
    EK = -100e-3
    ECa = 125e-3
    Em = -65e-3
    EAR = -40e-3
    conductance = {
	0: {
	    'NaF2':   0.4,
	    'KDR_FS':   0.4,
	    'KA':   0.002,
	    'K2':   0.0001
	    },
	1: {
	    'NaF2':   0.15,
	    'NaPF_SS':   0.00015,
	    'KDR_FS':   0.1,
	    'KC_FAST':   0.01,
	    'KA':   0.03,
	    'KM':   0.00375,
	    'K2':   0.0001,
	    'KAHP_SLOWER':   0.0001,
	    'CaL':   0.0005,
	    'CaT_A':   0.0001,
	    'AR':   0.00025
	    },
	2: {
	    'NaF2':   0.075,
	    'NaPF_SS':   7.5E-05,
	    'KDR_FS':   0.075,
	    'KC_FAST':   0.01,
	    'KA':   0.03,
	    'KM':   0.00375,
	    'K2':   0.0001,
	    'KAHP_SLOWER':   0.0001,
	    'CaL':   0.0005,
	    'CaT_A':   0.0001,
	    'AR':   0.00025
	    },
	3: {
	    'NaF2':   0.075,
	    'NaPF_SS':   7.5E-05,
	    'KDR_FS':   0.075,
	    'KC_FAST':   0.01,
	    'KA':   0.002,
	    'KM':   0.00375,
	    'K2':   0.0001,
	    'KAHP_SLOWER':   0.0001,
	    'CaL':   0.0005,
	    'CaT_A':   0.0001,
	    'AR':   0.00025
	    },
	4: {
	    'NaF2':   0.005,
	    'NaPF_SS':   5.E-06,
	    'KC_FAST':   0.01,
	    'KA':   0.002,
	    'KM':   0.00375,
	    'K2':   0.0001,
	    'KAHP_SLOWER':   0.0001,
	    'CaL':   0.0005,
	    'CaT_A':   0.0001,
	    'AR':   0.00025
	    },
	5: {
	    'NaF2':   0.005,
	    'NaPF_SS':   5.E-06,
	    'KA':   0.002,
	    'KM':   0.00375,
	    'K2':   0.0001,
	    'KAHP_SLOWER':   0.0001,
	    'CaL':   0.0005,
	    'CaT_A':   0.0001,
	    'AR':   0.00025
	    },
	6: {
	    'NaF2':   0.005,
	    'NaPF_SS':   5.E-06,
	    'KA':   0.002,
	    'KM':   0.00375,
	    'K2':   0.0001,
	    'KAHP_SLOWER':   0.0001,
	    'CaL':   0.0005,
	    'CaT_A':   0.0001,
	    'AR':   0.00025
	    },
	7: {
	    'NaF2':   0.005,
	    'NaPF_SS':   5.E-06,
	    'KA':   0.002,
	    'KM':   0.00375,
	    'K2':   0.0001,
	    'KAHP_SLOWER':   0.0001,
	    'CaL':   0.003,
	    'CaT_A':   0.0001,
	    'AR':   0.00025
	    },
	8: {
	    'NaF2':   0.005,
	    'NaPF_SS':   5.E-06,
	    'KA':   0.002,
	    'KM':   0.00375,
	    'K2':   0.0001,
	    'KAHP_SLOWER':   0.0001,
	    'CaL':   0.003,
	    'CaT_A':   0.0001,
	    'AR':   0.00025
	    },
	9: {
	    'NaF2':   0.005,
	    'NaPF_SS':   5.E-06,
	    'KA':   0.002,
	    'KM':   0.00375,
	    'K2':   0.0001,
	    'KAHP_SLOWER':   0.0001,
	    'CaL':   0.003,
	    'CaT_A':   0.0001,
	    'AR':   0.00025
	    }
	}

    channels = {'NaF2': 'NaF2_SS', 
                'NaPF_SS': 'NaPF_SS',
                'KDR_FS': 'KDR_FS',
                'KA': 'KA', 
                'K2': 'K2', 
                'KM': 'KM', 
                'KC_FAST': 'KC_FAST', 
                'KAHP_SLOWER': 'KAHP_SLOWER',
                'CaL': 'CaL', 
                'CaT_A': 'CaT_A', 
                'AR': 'AR'}

    def __init__(self, *args):
	moose.Cell.__init__(self, *args)
        self.channel_lib = {}
        for channel_class, channel_name in SpinyStellate.channels.items():
            channel = None
            if config.context.exists('/library/' + channel_name):
                channel = moose.HHChannel(channel_name, config.lib)
            else:
                class_obj = eval(channel_class)
                if channel_class == 'NaF2':
                    channel = class_obj(channel_name, config.lib, shift=-2.5e-3)
                else:
                    channel = class_obj(channel_name, config.lib)
            self.channel_lib[channel_class] = channel

	comp = []
	dendrites = set()
	level = defaultdict(set)
	axon = []
	self.presyn = 57
	for ii in range(60):
	    comp.append(MyCompartment('comp_' + str(ii), self))
        # Assign levels to the compartments
	level[ 1].add( comp[ 1]) 
	level[ 2].add( comp[ 2]) 
	level[ 3].add( comp[ 3]) 
	level[ 3].add( comp[ 4]) 
	level[ 4].add( comp[ 5]) 
	level[ 4].add( comp[ 6]) 
	level[ 4].add( comp[ 7]) 
	level[ 5].add( comp[ 8]) 
	level[ 5].add( comp[ 9]) 
	level[ 5].add( comp[ 10])
	level[ 6].add( comp[ 11])
	level[ 7].add( comp[ 12])
	level[ 8].add( comp[ 13])
	level[ 9].add( comp[ 14])
	level[ 2].add( comp[ 15])
	level[ 3].add( comp[ 16])
	level[ 3].add( comp[ 17])
	level[ 4].add( comp[ 18])
	level[ 4].add( comp[ 19])
	level[ 4].add( comp[ 20])
	level[ 5].add( comp[ 21])
	level[ 5].add( comp[ 22])
	level[ 5].add( comp[ 23])
	level[ 6].add( comp[ 24])
	level[ 7].add( comp[ 25])
	level[ 8].add( comp[ 26])
	level[ 9].add( comp[ 27])
	level[ 2].add( comp[ 28])
	level[ 3].add( comp[ 29])
	level[ 3].add( comp[ 30])
	level[ 4].add( comp[ 31])
	level[ 4].add( comp[ 32])
	level[ 4].add( comp[ 33])
	level[ 5].add( comp[ 34])
	level[ 5].add( comp[ 35])
	level[ 5].add( comp[ 36])
	level[ 6].add( comp[ 37])
	level[ 7].add( comp[ 38])
	level[ 8].add( comp[ 39])
	level[ 9].add( comp[ 40])
	level[ 2].add( comp[ 41])
	level[ 3].add( comp[ 42])
	level[ 3].add( comp[ 43])
	level[ 4].add( comp[ 44])
	level[ 4].add( comp[ 45])
	level[ 4].add( comp[ 46])
	level[ 5].add( comp[ 47])
	level[ 5].add( comp[ 48])
	level[ 5].add( comp[ 49])
	level[ 6].add( comp[ 50])
	level[ 7].add( comp[ 51])
	level[ 8].add( comp[ 52])
	level[ 9].add( comp[ 53])
	level[ 0].add( comp[ 54])
	level[ 0].add( comp[ 55])
	level[ 0].add( comp[ 56])
	level[ 0].add( comp[ 57])
	level[ 0].add( comp[ 58])
	level[ 0].add( comp[ 59])
	
	for ii in range(2, len(level)):
	    dendrites |= level[ii]
	self.level = level
	self.comp = comp
	self.dendrites = dendrites
	self.soma = comp[1]
	for compartment in comp[1:]:
	    compartment.length *= 1e-6
	    compartment.diameter *= 1e-6
	    compartment.setSpecificCm(9e-3)


        t1 = datetime.now()
	    
#	for ii in range(0, len(level)):
        for ii in range(2):
	    comp_set = level[ii]
	    conductances = SpinyStellate.conductance[ii]
	    mult = 1e4
	    if ii > 1:
		mult *= 2.0
	    for comp in comp_set:
                comp.Em = SpinyStellate.Em
                comp.initVm = SpinyStellate.Em
		for channel_name, density in conductances.items():
		    channel = moose.HHChannel(self.channel_lib[channel_name], 
					  channel_name, comp)
		    comp.insertChannel(channel, specificGbar=mult *  density)
                    if channel_name.startswith('K'):
                        channel.Ek = SpinyStellate.EK
                        if channel_name == 'KA' or channel_name == 'K2' or channel_name == 'KAHP_SLOWER' or channel_name == 'KDR_FS' or channel_name == 'KM':
                            channel.X = 0.0
                    elif channel_name.startswith('Na'):
                        channel.X = 0.0
                        channel.Ek = SpinyStellate.ENa
                    elif channel_name.startswith('Ca'):
                        channel.Ek = SpinyStellate.ECa
                    elif channel_name.startswith('AR'):
                        channel.Ek = SpinyStellate.EAR
                        channel.X = 0.0
                    else:
                        print 'ERROR: Unknown channel type:', channel
	for compartment in self.dendrites:
# 	    print compartment.name, compartment.length, compartment.diameter
	    compartment.setSpecificRm(5.0/2)
	    compartment.setSpecificRa(2.5)
	    compartment.Cm *= 2.0
	    compartment.insertCaPool(5.2e-6 / 2e-10, 20e-3)

	self.soma.setSpecificRm(5.0)
	self.soma.setSpecificRa(2.5)
        self.soma.insertCaPool(5.2e-6 / 2e-10, 50e-3)

	for compartment in self.level[0]: # axonal comps
	    compartment.setSpecificRm(0.1)
	    compartment.setSpecificRa(1.0)

        t2 = datetime.now()
        delta = t2 - t1
        print 'insert channels: ', delta.seconds + 1e-6 * delta.microseconds
		    

def has_cycle(comp):
    comp._visited = True
    ret = False
    for item in comp.raxial_list:
        if hasattr(item, '_visited') and item._visited:
            print 'Cycle between: ', comp.path, 'and', item.path
            return True
        ret = ret or has_cycle(item)
    return ret

def dump_cell(cell, filename):
    file_obj = open(filename, 'w')
    for lvl in cell.level:
        for comp in cell.level[lvl]:
            file_obj.write(str(lvl) + ' ' + comp.get_props() + '\n')
    file_obj.close()

    
import pylab
from simulation import Simulation
import pymoose

if __name__ == '__main__':
    sim = Simulation()
    s = SpinyStellate('cell', sim.model)
    vm_table = s.soma.insertRecorder('Vm', sim.data)
    pulsegen = s.soma.insertPulseGen('pulsegen', sim.model, firstLevel=3e-10, firstDelay=20e-3, firstWidth=100e-3)
    sim.schedule()
    if has_cycle(s.soma):
        print "WARNING!! CYCLE PRESENT IN CICRUIT."

    t1 = datetime.now()
    sim.run(100e-3)
    t2 = datetime.now()
    delta = t2 - t1
    print 'simulation time: ', delta.seconds + 1e-6 * delta.microseconds
    sim.dump_data('data')
    dump_cell(s, 'brutess.txt')
    pylab.plot(vm_table)
    pylab.show()

# 
# brutespinstell.py ends here
