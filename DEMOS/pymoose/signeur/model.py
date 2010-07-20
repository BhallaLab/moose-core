# model.py --- 
# 
# Filename: model.py
# Description: 
# Author: Gael Jalowicki and Subhasis Ray
# Maintainer: 
# Created: Sat Jul 17 11:57:39 2010 (+0530)
# Version: 
# Last-Updated: Tue Jul 20 19:01:35 2010 (+0530)
#           By: subha
#     Update #: 471
# URL: 
# Keywords: multiscale model, signaling, compartmental model, neuroinformatics, systems biology
# Compatibility: 
# 
# It requires matplotlib and moose.

# Commentary: 
# 
# This script demonstrates the use of pymoose for simulating a
# multiscale model combining chemical kinetics model and biophysical
# neuronal cell model. 
# 
# The neuronal model is a simplified version of a hippocampal CA1
# pyramidal cell (genesis/simpleca1.p). It has a soma, one apical
# dendrite with 10 compartments and two of the dendritic compartments
# (#6 and #9) have two spines: each consisting of a neck compartment
# and a head compartment.
#
# We have two kinetic models in SBML: kinase_loop in dendrite
# (sbml/kinase_loop.xml) and psd12 (sbml/psd12.xml) in spines.  The
# [Ca2+] in the compartments change due to Ca+2 currents when the
# neuron fires. This is propagated to the signaling models and the
# amount of products of signaling pathway follow the firing behaviour
# of the neuron.

# Change log:
# 
# Initial working version developed by subha based on scripts expt.g
# by Upi and testSigNeurSimpleCa1.py by Gael.
# 
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 
# 

# Code:

import sys
import os

import pylab
import numpy
import logging

import moose
import pymoose

def handleError(self, record):
    raise

class config(object):
    SIMDT = 50.0e-6
    SETTLE_TIME = 2.0
    RUNTIME = 10.0
    CELLDT = 50.0e-6
    SIGDT = 5.0e-4
    SIGPLOTDT = 5.0e-3
    CA_SCALING = 0.06
    LOG_FILENAME = sys.stdout
    LOG_LEVEL = logging.DEBUG
    logging.Handler.handleError = handleError
    logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s %(levelname)s %(name)s %(filename)s %(funcName)s: %(message)s', filemode='w')
    LOGGER = logging.getLogger('ca1')

    context = moose.PyMooseBase.getContext()
    sbml_dir = 'sbml'
    genesis_dir = 'genesis'
    protocol_dir = 'protocol'

class Model(object):
    """Wrap moose model for Ca1 with Signalling
    and Neuronal submodels.

    """
    def __init__(self, *args):
	object.__init__(self, *args)
	self._model_files = {}
	# Each file path as a member variable
	self._model_files['proto16'] = os.path.join(config.genesis_dir, 'proto16.g')
	self._model_files['kinase_loop'] = os.path.join(config.sbml_dir, 'kinase_loop.xml')
	self._model_files['dend'] = os.path.join(config.genesis_dir, 'dend_v27.g')
	self._model_files['diffmol_dend'] = os.path.join(config.genesis_dir, 'diffmol_dend.g')
	self._model_files['psd12'] = os.path.join(config.sbml_dir, 'psd12.xml')
	self._model_files['diffmol_spine'] = os.path.join(config.genesis_dir, 'diffmol_spine.g')
        self._model_files['simpleca1'] = os.path.join(config.genesis_dir, 'simpleca1.p')
        self._model_files['pulse_input'] = os.path.join(config.protocol_dir, 'pulse_LTP_protocol2.txt')

	# Check that all the model files are accessible
	for filename in self.missing_model_files():
	    print 'ERROR: could not access', filename
        config.context.runG('float DEFAULT_VOL = 1.257e-16')
        config.context.runG('float PSD_VOL = 1.0e-20')
            
	# Library /library is a global
	self.lib = moose.Neutral('/library')
        self.data = moose.Neutral('/data')
        self.make_dend()
        print 'After dend'
        pymoose.printtree('/')
        self.make_spine()
        print 'After spine'
        pymoose.printtree('/')
        self.make_cell()
        print 'After readcell'
        pymoose.printtree(moose.Neutral('/'))
        self.make_signeur()
        print 'After signeur'
        pymoose.printtree(moose.Neutral('/'))
        print 'Showing messages for cell/spine/[Ca2+]'
        pymoose.showmsg('/sig/cell/spine_head_14_1/NMDA_Ca_conc')
        print 'Showing messages for kinetics/spine/[Ca]'
        pymoose.showmsg('/sig/kinetics/spine[0]/A_67_0_/Ca_81_0_')
        
        self.glu_1 = moose.SynChan('/sig/cell/spine_head_14_1/glu')
        self.pulse_input = moose.TimeTable('/pulse')
        self.pulse_input.method = 4
        self.pulse_input.filename = self._model_files['pulse_input']
        self.pulse_input.connect('event', self.glu_1, 'synapse')
        self.glu_1.setWeight(0, 1.0)
        self.setup_recording()        

    def make_dend(self):
	"""Load the dendrite model"""
	self.kinetics = moose.KinCompt('/kinetics')
	self.kinetics.size = 1e-15
# 	config.context.loadG(self._model_files['dend']) # The den_27.g is replaced by kinase_loop
	config.context.readSBML(self._model_files['kinase_loop'], '/kinetics')
	config.context.loadG(self._model_files['diffmol_dend'])
	config.context.move(self.kinetics.id, self.lib.id, 'dend')
        self.dend_proto_path = self.lib.path + '/dend'
        
    def make_spine(self):
	"""Load the spine model"""
	self.kinetics = moose.KinCompt('/kinetics')
	self.kinetics.size = 1e-15
	config.context.readSBML(self._model_files['psd12'], '/kinetics')
	config.context.loadG(self._model_files['diffmol_spine'])
	config.context.move(self.kinetics.id, self.lib.id, 'spine')
        self.spine_proto_path = self.lib.path + '/spine'

    def make_cell(self):
	self.make_proto16()
        config.context.readCell(self._model_files['simpleca1'], '/library/cell')
        self.cell_proto_path =  self.lib.path + '/cell'
        self.cell = moose.Cell(self.cell_proto_path)

    def make_proto16(self):
	# Load the files and move the model components around
	config.context.loadG(self._model_files['proto16'])
	current = config.context.getCwe()
	config.context.setCwe(self.lib.id)
	config.context.runG('create symcompartment symcompartment')
	config.context.runG('make_Na')
	config.context.runG('make_Ca')
	config.context.runG('make_K_DR')
	config.context.runG('make_K_AHP')
	config.context.runG('make_K_C')
	config.context.runG('make_K_A')
	config.context.runG('make_Ca_conc')
	config.context.runG('make_glu')
	config.context.runG('make_NMDA')
	config.context.runG('make_Ca_NMDA')
	config.context.runG('make_NMDA_Ca_conc')    
	config.context.setCwe(current)

    def make_signeur(self):
        self.signeur = moose.SigNeur('/sig')
        self.signeur.cellProto = self.cell_proto_path
        self.signeur.dendProto = self.dend_proto_path
        self.signeur.spineProto = self.spine_proto_path
        self.signeur.lambda_ = 5e-5
        self.signeur.Dscale = 1e-12
        config.context.runG('setfield %s calciumMap[NMDA_Ca_conc] Ca_81_0_' % (self.signeur.path)) # To spine
        config.context.runG('setfield %s calciumMap[Ca_conc] Ca_61_0_' % (self.signeur.path)) # To dend
        config.context.runG('setfield %s calciumMap[K_A] K_A' % (self.signeur.path)) # dend to cell
        self.signeur.sigDt = config.SIGDT
        self.signeur.cellDt = config.CELLDT
        self.signeur.calciumScale = config.CA_SCALING
        self.signeur.dendInclude = 'apical'
        self.signeur.dendExclude = ''
        print 'Building SigNeur'
        self.signeur.build()
        print 'Finished building SigNeur'
    
    def setup_recording(self):
        """Set up data tables for recording state variables from the model

        """
        self.data_tables = []

        # Record the input pulse
        table = moose.Table('pulse', self.data)
        table.stepMode = 3
        table.connect('inputRequest', self.pulse_input, 'state')
        self.data_tables.append(table)

        # Record spine Ca-pool This causes a segmentation fault. There
        # is a bug in Table that causes this. When multiple tables
        # connect to the same field, the messaging gets corrupted
        # moose gets a segmentation fault. SigNeur connects an adpater
        # table to NMDA_Ca_conc and hence we cannot connect a
        # recording table to it without causing a crash.


#         if config.context.exists('/sig/cell/spine_head_14_1/NMDA_Ca_conc'):
#             table = moose.Table('SpineCaPool1', self.data)
#             table.stepMode = 3
#             table.connect('inputRequest', moose.Neutral('/sig/cell/spine_head_14_1/NMDA_Ca_conc'), 'Ca')
#             print "###" , moose.Neutral('/sig/cell/spine_head_14_1/NMDA_Ca_conc').className
#             pymoose.showmsg('/sig/cell/spine_head_14_1/NMDA_Ca_conc')
#             self.data_tables.append(table)
#         else:
#             print '/sig/cell/spine_head_14_1/NMDA_Ca_conc does not exist'
        
        # Record Vm in soma
        if config.context.exists('/sig/cell/spine_head_14_1'):
            table = moose.Table('SpineVm', self.data)
            table.stepMode = 3
            table.connect('inputRequest', moose.Neutral('/sig/cell/spine_head_14_1'), 'Vm')
            self.data_tables.append(table)
        else:
            print '/sig/cell/spine_head_14_1 does not exist'

        if config.context.exists('/sig/kinetics/spine[0]/A_67_0_/M_star__98_0_'):
            table = moose.Table('MStarA', self.data)
            table.stepMode = 3
            table.connect('inputRequest', moose.Neutral('/sig/kinetics/spine[0]/A_67_0_/M_star__98_0_'), 'n')            
            self.data_tables.append(table)
        else:
            print '/sig/kinetics/spine[0]/A_67_0_/M_star__98_0_ does not exist'
        if config.context.exists('/sig/kinetics/spine[0]/A_67_0_/Ca_81_0_'):
            table = moose.Table('SpineCa', self.data)
            table.stepMode = 3
            table.connect('inputRequest', moose.Neutral('/sig/kinetics/spine[0]/A_67_0_/Ca_81_0_'), 'n')            
            self.data_tables.append(table)
        else:
            print '/sig/kinetics/spine[0]/A_67_0_/Ca_81_0_ does not exist'

        return self.data_tables

    def run(self, runtime):
        """Set clocks, reset and run the simulation

        """
        config.context.setClock(0, config.SIMDT)
        config.context.setClock(1, config.SIMDT)
        config.context.setClock(2, config.SIMDT)
        config.context.setClock(3, config.SIMDT)
        config.context.setClock(4, config.SIGPLOTDT)
        config.context.useClock(4, self.data.path + '/#[TYPE=Table]')
        print 'Reset 1'
        config.context.reset()
        print 'Reset 2'
        config.context.reset()
        print 'Starting simulation'
        config.context.step(config.SETTLE_TIME)
        print 'Settled simulation'
        config.context.step(runtime)
        print 'Finished running'

    def plot_data(self):
        """Plot the recorded data
        
        """
        plot_count = len(self.data_tables)
        rows = 2
        cols = plot_count / rows
        while rows * cols < plot_count:
            cols += 1
        time = numpy.linspace(0, config.context.getCurrentTime(), len(self.data_tables[-1]))
        
        index = 1
        for table in self.data_tables:
            table.dumpFile(table.name + '.plot')
            data = numpy.array(table)
            print 'Data is of size: ', len(data)
            pylab.subplot(rows, cols, index)
            pylab.plot(time, data, label=table.name)            
            pylab.legend()
            index += 1
        pylab.show()
        
    def missing_model_files(self):
	ret = []
	for key, filename in self._model_files.items():
	    if not os.access(filename, os.R_OK):
		ret.append(filename)
	return ret


def test_model():
    model = Model()
    print 'Going to run'
    model.run(config.RUNTIME)
    model.plot_data()
    print 'Finished simulation'

if __name__ == '__main__':
    test_model()

# 
# model.py ends here
