# ss.py --- 
# 
# Filename: ss.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Mon May 18 02:22:10 2009 (+0530)
# Version: 
# Last-Updated: Thu May 21 02:15:55 2009 (+0530)
#           By: subhasis ray
#     Update #: 306
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

    # Radius mapped to tree node no.
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
    # The lengths are mapped to levels
    length = {0: 50.0,
	      1: 20.0,
	      2: 40.0,
	      3: 40.0,
	      4: 40.0,
	      5: 40.0,
	      6: 40.0,
	      7: 40.0,
	      8: 40.0,
	      9: 40.0}

    ENa = 50e-3
    EK = -100e-3
    ECa = 125e-3
    Em = -65e-3
    EAR = -40e-3

    comp_cnt = 0

    channels = {'NaF2': 'NaF2_SS', 
                'NaPF_SS': 'NaPF_SS',
                'KDR_FS': 'KDR_FS_SS',
                'KA': 'KA_SS', 
                'K2': 'K2_SS', 
                'KM': 'KM_SS', 
                'KC_FAST': 'KC_FAST_SS', 
                'KAHP_SLOWER': 'KAHP_SLOWER_SS',
                'CaL': 'CaL_SS', 
                'CaT_A': 'CaT_A_SS', 
                'AR': 'AR_SS'}

    channel_density = {	
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


    def __init__(self, *args):
	moose.Cell.__init__(self, *args)
	self.channels_inited = False
	self.channel_lib = {}
	self._init_channels()
	self.level = defaultdict(set) # Python >= 2.5 
	self.dendrites = set() # List of compartments that are not
				 # part of axon
	self.axon = []
	self._create_comps()
        self._set_passive()
        self._connect_axial(self.soma)
        self._insert_channels()

    def _add_comp(self, parent, level_no, dia=15e-6, length=40e-6):
	SpinyStellate.comp_cnt += 1
	comp = MyCompartment('comp_' + str(SpinyStellate.comp_cnt), parent)
        self.level[level_no].add(comp)
        comp.diameter = dia
        comp.length = length
        return comp

    def _create_comps(self):
	t1 = datetime.now()
	self.soma = self._add_comp(self, 1, dia=SpinyStellate.radius[0] * 2e-6,
                                   length=SpinyStellate.length[1] * 1e-6)
	for i in range(4):
	   self._create_dtree(self.soma, SpinyStellate.dendritic_tree, 2)

        self._create_axon()
	t2 = datetime.now()
        delta = t2 - t1
        print '_create_comps took: ', delta.seconds + 1e-6 * delta.microseconds

    def _create_dtree(self, parent, tree, level, radius_dict=radius):
	"""Create the dendritic tree structure with compartments."""
        if not tree:
            return
        comp = self._add_comp(parent, 
                              level, 
                              dia=radius_dict[tree[0]] * 2e-6,
                              length=SpinyStellate.length[level] * 1e-6)
        self.dendrites.add(comp)
        for subtree in tree[1:]:
            self._create_dtree(comp, subtree, level+1, radius_dict)

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
        level_no = 0
        a0 = self._add_comp(self.soma, 
                            level_no, 
                            dia=0.7 * 2e-6, 
                            length=SpinyStellate.length[level_no] * 1e-6)
        self.axon.append(a0)
        a1 = self._add_comp(a0, 
                            level_no, 
                            dia=0.6 * 2e-6, 
                            length=SpinyStellate.length[level_no] * 1e-6)
        self.axon.append(a1)
        a00 = self._add_comp(a1, 
                             level_no, 
                             dia=0.5 * 2e-6, 
                             length=SpinyStellate.length[level_no] * 1e-6)
        self.axon.append(a00)
        a01 = self._add_comp(a00, 
                             level_no, 
                             dia=0.5 * 2e-6, 
                             length=SpinyStellate.length[level_no] * 1e-6)
        self.axon.append(a01)
        a10 = self._add_comp(a1, 
                             level_no, 
                             dia=0.5 * 2e-6, 
                             length=SpinyStellate.length[level_no] * 1e-6)
        self.axon.append(a10)
        a11 = self._add_comp(a10, 
                             level_no, 
                             dia=0.5 * 2e-6, 
                             length=SpinyStellate.length[level_no] * 1e-6)
        self.axon.append(a11)
        a11 = self._add_comp(a10, 
                             level_no, 
                             dia=0.5 * 2e-6, 
                             length=SpinyStellate.length[level_no] * 1e-6)
        self.axon.append(a11)

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
	    if isinstance(channel, KC_FAST):
		channel.Z = 0.0
	    else:
		channel.X = 0.0
# 	    channel.X = 0.0
	    if isinstance(channel, KChannel):
		channel.Ek = SpinyStellate.EK
	    elif isinstance(channel, NaChannel):
		channel.Ek = SpinyStellate.ENa
	    elif isinstance(channel, CaChannel):
		channel.Ek = SpinyStellate.ECa
	    elif isinstance(channel, AR):
		channel.Ek = SpinyStellate.EAR
	    else:
		raise Exception, 'Unknown channel type:' + channel.__class__.__name__
            self.channel_lib[channel_class] = channel

        self.channels_inited = True
	
    def _set_passive(self):
	self.soma.setSpecificCm(9e-3)
	self.soma.setSpecificRm(5.0)
	self.soma.setSpecificRa(2.5)
        self.soma.Em = SpinyStellate.Em
        self.soma.initVm = SpinyStellate.Em
	for comp in self.dendrites:
	    comp.setSpecificCm(9e-3 * SpinyStellate.spine_area_mult)
	    comp.setSpecificRm(5.0 / SpinyStellate.spine_area_mult)
	    comp.setSpecificRa(2.5)
            comp.Em = SpinyStellate.Em
            comp.initVm = SpinyStellate.Em
            print comp.get_props()

	for comp in self.axon:
	    comp.setSpecificCm(9e-3)
	    comp.setSpecificRm(0.1)
	    comp.setSpecificRa(1.0)
            comp.Em = SpinyStellate.Em
            comp.initVm = SpinyStellate.Em

    def _insert_channels(self):
        if not self.channels_inited:
            raise Exception, 'Channels not initialized in library'

        t1 = datetime.now()
        for level_no, comp_set in self.level.items():
            mult = 1.0
            if level_no > 1:
                mult = SpinyStellate.spine_area_mult
	    conductances = SpinyStellate.channel_density[level_no]
	    for comp in comp_set:
		for channel_name, density in conductances.items():
		    channel = moose.HHChannel(self.channel_lib[channel_name], 
					  channel_name, comp)
		    comp.insertChannel(channel, specificGbar=mult * density * 1e4)
	for comp in self.dendrites:
	    comp.insertCaPool(5.2e-6 / 2e-10, 20e-3)
	self.soma.insertCaPool(5.2e-6 / 2e-10, 50e-3)
        t2 = datetime.now()
        delta = t2 - t1
        print 'insert channels: ', delta.seconds + 1e-6 * delta.microseconds

    def _connect_axial(self, root):
        """Connect parent-child compartments via axial-raxial
        messages."""
        parent = moose.Neutral(root.parent)
        
        if parent.className == 'Compartment' and not hasattr(root, 'axial_connected'):
            root.connect('raxial', parent, 'axial')
            root.axial_connected = True
            print root.name, parent.name

        for child in root.children():
            obj = moose.Neutral(child)
            if obj.className == 'Compartment':
                self._connect_axial(obj)

def dump_cell(cell, filename):
    file_obj = open(filename, 'w')
    for lvl in cell.level:
        for comp in cell.level[lvl]:
            file_obj.write(str(lvl) + ' ' + comp.get_props() + '\n')
    file_obj.close()

from datetime import datetime
import pylab
from simulation import Simulation

if __name__ == '__main__':
    sim = Simulation()
    t1 = datetime.now()
    s = SpinyStellate('ss', sim.model)
    t2 = datetime.now()
    delta_t = t2 - t1
    print '### TIME SPENT IN CELL CREATION: ', delta_t.seconds + delta_t.microseconds * 1e-6
#    pymoose.printtree(s.soma)
    dump_cell(s, 'ss.p')
#     path = s.soma.path + '/a_0/a_1/a_0_0/a_0_1'
#     a2 = MyCompartment(path)
    vm_table = s.soma.insertRecorder('Vm', sim.data)
    s.soma.insertPulseGen('pulsegen', sim.model, firstLevel=3e-10, firstDelay=20e-3, firstWidth=50e-3)
    sim.schedule()
    t1 = datetime.now()
    sim.run(50e-3)
    t2 = datetime.now()
    delta_t = t2 - t1
    print '#### TIME TO SIMULATE:', delta_t.seconds + delta_t.microseconds * 1e-6
    sim.dump_data('data')
    nrn_data = pylab.loadtxt('../nrn/mydata/Vm_ss.plot')
    nrn_vm = nrn_data[:, 1]
    nrn_t = nrn_data[:, 0]
    mus_vm = pylab.array(vm_table) * 1e3 # convert Neuron unit - mV
    mus_t = pylab.linspace(0, sim.simtime * 1e3, len(vm_table)) # convert simtime to neuron unit - ms
    pylab.plot(mus_t, mus_vm, 'r-', label='mus')
    pylab.plot(nrn_t, nrn_vm, 'g-', label='nrn')
    pylab.legend()
    pylab.show()



# 
# ss.py ends here
