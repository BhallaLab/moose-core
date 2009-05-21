# spinystellate.py --- 
# 
# Filename: spinystellate.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Wed Apr 29 10:24:37 2009 (+0530)
# Version: 
# Last-Updated: Thu May 21 01:39:39 2009 (+0530)
#           By: subhasis ray
#     Update #: 571
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Initial version of a SpinyStellate cell implementation. Will
# refactor into a base class for Cell and then subclass here once I
# get the hang of the commonalities.
# 
# 

# Change log:
# 
# 
# 
# 
# Code:
from collections import deque, defaultdict
import moose

from kchans import *
from nachans import *
from cachans import *
from capool import *
from archan import *

from compartment import MyCompartment


class SpinyStellate(moose.Cell):
    spine_area_mult = 2.0 # accomodates apine area
    #
    #     The dendritic structure is like this:
    #
    # level:    2  3  4  5   6  7   8    9
    #           c1-c2-c4-c7
    #            \
    #              c3-c5-c8
    #               \
    #                 c6-c9-c10-c11-c12-c13
    #
    # The compartments are numbered 1 to 13 and then we traverse the
    # tree in a breadth first manner.
    #
    # The following list captures the structure of the dendrites.
    # it is like:
    # subtree ::= [ node, child_subtree1, child_subtree2, child_subtree3, ...]
    dendritic_tree = [1, [2, [ 4, [7] ] ], 
                      [3, [5, [8]], 
                       [6, [9, [10, [11, [12, [13]]]]]]]]
    
    # radius in microns - the keys are the dendritic compartment
    # identities in dendritic_tree
    radius = {0: 7.5, # soma - actually this will be compartment no. 1
	      1: 1.06, 
	      2: 0.666666667,
	      3: 0.666666667, 
	      4: 0.666666667, 
	      5: 0.418972332, 
	      6: 0.418972332, 
	      7: 0.666666667, 
	      8: 0.418972332, 
	      9: 0.418972332, 
	      10: 0.418972332, 
	      11: 0.418972332, 
	      12: 0.418972332, 
	      13: 0.418972332} 
    ENa = 50e-3
    EK = -100e-3
    ECa = 125e-3
    Em = -65e-3
    EAR = -40e-3
    channel_density = {0: {'NaF2':     4000.0,
                           'KDR_FS':   4000.0,
                           'KA':       20.0,
                           'K2':       1.0},

                       1: {'NaF2':     1500.0,
                           'NaPF_SS':  1.5,
                           'KDR_FS':   1000.0,
                           'KC_FAST':  100.0,
                           'KA':       300.0,
                           'KM':       37.5,
                           'K2':       1.0,
                           'KAHP_SLOWER':      1.0,
                           'CaL':      5.0,
                           'CaT_A':    1.0,
                           'AR':       2.5},

                       2:{'NaF2':      750.0,
                          'NaPF_SS':   0.75,
                          'KDR_FS':    750.0,
                          'KC_FAST':   100.0,
                          'KA':        300.0,
                          'KM':        37.5,
                          'K2':        1.0,
                          'KAHP_SLOWER':       1.0,
                          'CaL':       5.0,
                          'CaT_A':     1.0,
                          'AR':        2.5},

                       3: {'NaF2':     750.0,
                           'NaPF_SS':  0.75,
                           'KDR_FS':   750.0,
                           'KC_FAST':  100.0,
                           'KA':       20.0,
                           'KM':       37.5,
                           'K2':       1.0,
                           'KAHP_SLOWER': 1.0,
                           'CaL':      5.0,
                           'CaT_A':    1.0,
                           'AR':       2.5},
                       
                       4: {'NaF2': 0.005 * 1e4,
                           'NaPF_SS': 5.E-06 * 1e4,
                           'KC_FAST': 0.01 * 1e4,
                           'KA': 0.002 * 1e4,
                           'KM': 0.00375 * 1e4,
                           'K2': 0.0001 * 1e4,
                           'KAHP_SLOWER': 0.0001 * 1e4,
                           'CaL': 0.0005 * 1e4,
                           'CaT_A': 0.0001 * 1e4,
                           'AR': 0.00025 * 1e4},
                       
                       5: {'NaF2': 0.005 * 1e4,
                           'NaPF_SS': 5.E-06 * 1e4,
                           'KA': 0.002 * 1e4,
                           'KM': 0.00375 * 1e4,
                           'K2': 0.0001 * 1e4,
                           'KAHP_SLOWER': 0.0001 * 1e4,
                           'CaL': 0.0005 * 1e4,
                           'CaT_A': 0.0001 * 1e4,
                           'AR': 0.00025 * 1e4},

                       6: {'NaF2': 0.005 * 1e4,
                           'NaPF_SS': 5.E-06 * 1e4,
                           'KA': 0.002 * 1e4,
                           'KM': 0.00375 * 1e4,
                           'K2': 0.0001 * 1e4,
                           'KAHP_SLOWER': 0.0001 * 1e4,
                           'CaL': 0.0005 * 1e4,
                           'CaT_A': 0.0001 * 1e4,
                           'AR': 0.00025 * 1e4},

                       7: {'NaF2': 0.005 * 1e4, 
                           'NaPF_SS': 5.E-06 * 1e4, 
                           'KA': 0.002 * 1e4, 
                           'KM': 0.00375 * 1e4, 
                           'K2': 0.0001 * 1e4, 
                           'KAHP_SLOWER': 0.0001 * 1e4, 
                           'CaL': 0.003 * 1e4, 
                           'CaT_A': 0.0001 * 1e4, 
                           'AR': 0.00025 * 1e4},

                       8: {'NaF2': 0.005 * 1e4, 
                           'NaPF_SS': 5.E-06 * 1e4, 
                           'KA': 0.002 * 1e4, 
                           'KM': 0.00375 * 1e4, 
                           'K2': 0.0001 * 1e4, 
                           'KAHP_SLOWER': 0.0001 * 1e4, 
                           'CaL': 0.003 * 1e4, 
                           'CaT_A': 0.0001 * 1e4, 
                           'AR': 0.00025 * 1e4},

                       9: {'NaF2': 0.005 * 1e4,
                           'NaPF_SS': 5.E-06 * 1e4,
                           'KA': 0.002 * 1e4,
                           'KM': 0.00375 * 1e4,
                           'K2': 0.0001 * 1e4,
                           'KAHP_SLOWER': 0.0001 * 1e4,
                           'CaL': 0.003 * 1e4,
                           'CaT_A': 0.0001 * 1e4,
                           'AR': 0.00025 * 1e4}}

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
        self.channels_inited = False
        self.channel_lib = {}
        self._init_channels()
        self.comp = [] # Keep a list of compartments for debugging
	self.levels = defaultdict(set) # Python >= 2.5 
	self.dendrites = set() # List of compartments that are not
				 # part of axon
	self.axon = []
        self._create_cell()
        self._set_passiveprops()
        self._connect_axial(self.soma)
        self._insert_channels()
        self.soma.insertCaPool(5.2e-6 / 2e-10, 50e-3)
        for comp in self.dendrites:
            comp.insertCaPool(5.2e-6 / 2e-10, 20e-3)

    def _init_channels(self):
        if self.channels_inited:
            return
        
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
            channel.X = 0.0
            self.channel_lib[channel_class] = channel
        self.channels_inited = True

    def _create_axon(self):
        """Create the axonal structure.

        It is like:       
                          a_0_0 -- a_0_1
                         /
                        /
        soma -- a_0 -- a_1
                       \
                        \
                         a_1_0 --  a_1_1
        """
        self.axon.append(MyCompartment('a_0', self.soma))
        self.axon[-1].diameter = 0.7 * 2e-6
        self.axon.append(MyCompartment('a_1', self.axon[0]))
        self.axon[-1].diameter = 0.6 *  2e-6
        self.axon.append(MyCompartment('a_0_0', self.axon[1]))
        self.axon.append(MyCompartment('a_1_0', self.axon[1]))
        self.axon.append(MyCompartment('a_0_1', self.axon[2]))
        self.axon.append(MyCompartment('a_1_1', self.axon[3]))
        for comp in self.axon[2:]: comp.diameter = 0.5 * 2e-6
        for comp in self.axon: 
            self.levels[0].add(comp)
            comp.length = 50e-6
        self.comp += self.axon

    def _create_dtree(self, name_prefix, parent, tree, level, default_length=40e-6, radius_dict=radius):
	"""Create the dendritic tree structure with compartments.

	Returns the root."""
        if not tree:
            return
        comp = MyCompartment(name_prefix + str(tree[0]), parent)
        self.comp.append(comp)
        comp.length = default_length
        comp.diameter = radius_dict[tree[0]] * 2e-6
        self.levels[level].add(comp)
        self.dendrites.add(comp)
        for subtree in tree[1:]:
            self._create_dtree(name_prefix, comp, subtree, level+1, default_length, radius_dict)
        

    def _create_cell(self):
        """Create the compartmental structure and set the geometry."""
	if not hasattr(self, 'levels'):
	    self.levels = defaultdict(set)
	comp = MyCompartment('soma', self)
        self.comp.append(comp)
	comp.length = 20e-6
	comp.diameter = 7.5 * 2e-6
	self.soma = comp
	self.levels[1].add(comp)
        t1 = datetime.now()
	for i in range(4):
	   self. _create_dtree('d_' + str(i) + '_', comp, SpinyStellate.dendritic_tree, 2)
        t2 = datetime.now()
        delta = t2 - t1
        print 'create_dtree took: ', delta.seconds + 1e-6 * delta.microseconds
#         self._create_axon()

    def _set_passiveprops(self):
        """Set the passive properties of the cells."""
        self.soma.setSpecificCm(9e-3)
        self.soma.setSpecificRm(5.0)
        self.soma.setSpecificRa(2.5)
        self.soma.Em = SpinyStellate.Em
        self.soma.initVm = SpinyStellate.Em
        for comp in self.dendrites:
            comp.setSpecificCm(9e-3 * SpinyStellate.spine_area_mult)
            comp.setSpecificRm(5.0/SpinyStellate.spine_area_mult)
            comp.setSpecificRa(2.5)
            comp.Em = SpinyStellate.Em
            comp.initVm = SpinyStellate.Em
        for comp in self.axon:
            comp.setSpecificCm(9e-3)
            comp.setSpecificRm(0.1)
            comp.setSpecificRa(1.0)
            comp.Em = SpinyStellate.Em
            comp.initVm = SpinyStellate.Em

    def _connect_axial(self, root):
        """Connect parent-child compartments via axial-raxial
        messages."""
        parent = moose.Neutral(root.parent)
        if parent.className == 'Compartment' and not hasattr(root, 'axial_connected'):
            root.connect('raxial', parent, 'axial')
            root.axial_connected = True
        
        for child in root.children():
            obj = moose.Neutral(child)
            if obj.className == 'Compartment':
                self._connect_axial(obj)

    def _insert_channels(self):
        if not self.channels_inited:
            raise Exception, 'Channels not initialized in library'

        t1 = datetime.now()
        for level in range(10):
            comp_set = self.levels[level]
            mult = 1.0
            if level > 1:
                mult = SpinyStellate.spine_area_mult
                
            for comp in comp_set:
                for channel, density in SpinyStellate.channel_density[level].items():
                    chan = moose.HHChannel(self.channel_lib[channel], channel, comp) # this does a copy
                    comp.insertChannel(chan, specificGbar=mult * density)
                    if channel.startswith('K'):
                        chan.X = 0.0
                        chan.Ek = SpinyStellate.EK
                    elif channel.startswith('Na'):
                        chan.X = 0.0
                        chan.Ek = SpinyStellate.ENa
                    elif channel.startswith('Ca'):
                        chan.X = 0.0
                        chan.Ek = SpinyStellate.ECa
                    elif channel.startswith('AR'):
                        chan.Ek = SpinyStellate.EAR
                        chan.X = 0.0
                    else:
                        print 'ERROR: Unknown channel type:', channel
        t2 = datetime.now()
        delta = t2 - t1
        print 'insert channels: ', delta.seconds + 1e-6 * delta.microseconds

def dump_cell(cell, filename):
    file_obj = open(filename, 'w')
    for lvl in cell.levels:
        for comp in cell.levels[lvl]:
            file_obj.write(str(lvl) + ' ' + comp.get_props() + '\n')
    file_obj.close()


#import pylab
import pymoose
from simulation import Simulation

#import timeit
from datetime import datetime
import pylab
if __name__ == '__main__':
    sim = Simulation()
    t1 = datetime.now()
    s = SpinyStellate('ss', sim.model)
    t2 = datetime.now()
    delta_t = t2 - t1
    print '### TIME SPENT IN CELL CREATION: ', delta_t.seconds + delta_t.microseconds * 1e-6
#    pymoose.printtree(s.soma)
    dump_cell(s, 'ss.p')
    path = s.soma.path + '/a_0/a_1/a_0_0/a_0_1'
    a2 = MyCompartment(path)
    vm_table = s.soma.insertRecorder('Vm', sim.data)
    s.soma.insertPulseGen('pulsegen', sim.model, firstLevel=3e-10, firstDelay=20e-3, firstWidth=100e-3)
    sim.schedule()
    t1 = datetime.now()
    sim.run(100e-3)
    t2 = datetime.now()
    delta_t = t2 - t1
    print '#### TIME TO SIMULATE:', delta_t.seconds + delta_t.microseconds * 1e-6
    sim.dump_data('data')
    pylab.plot(vm_table)
    pylab.show()
    
# 
# spinystellate.py ends here
