# lif.py --- 
# 
# Filename: lif.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sun Jul  8 14:00:31 2012 (+0530)
# Version: 
# Last-Updated: Sun Jul  8 15:06:29 2012 (+0530)
#           By: subha
#     Update #: 126
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Demonstrates use of Leaky Integrate and Fire (LeakyIaf class) in
# moose.
# 
# 

# Change log:
# 
# 
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
sys.path.append('../../python')
import os
os.environ['NUMPTHREADS'] = '1'

import moose
from matplotlib import pyplot as plt
import numpy as np

def setupmodel(modelpath, iaf_Rm, iaf_Cm, pulse_interval):
    """Create a LeakyIaF neuron under `modelpath` and a synaptic
    channel (SynChan) in it. Create a spike generator stimulated by a
    pulse generator to give input to the synapse.
    """
    model_container = moose.Neutral(modelpath)
    data_container = moose.Neutral(datapath)
    iaf = moose.LeakyIaF('%s/iaf' % (modelpath))
    iaf.Rm = iaf_Rm
    iaf.Cm = iaf_Cm
    iaf.initVm = -65
    iaf.Em = -65
    iaf.Vreset = -65
    iaf.Vthreshold = -40
    syn = moose.SynChan('%s/syn' % (iaf.path))
    syn.synapse.num = 1
    syn.Ek = -65
    syn.Gk = 1.0
    moose.connect(syn, 'IkOut', iaf, 'injectDest')
    moose.connect(iaf, 'VmOut', syn, 'Vm')
    sg = moose.SpikeGen('%s/spike' % (modelpath))
    sg.threshold = 0.5
    moose.connect(sg, 'event', syn.synapse[0], 'addSpike')
    pg = moose.PulseGen('%s/pulse' % (modelpath))
    pg.delay[0] = pulse_interval
    pg.width[0] = 1e-3
    pg.level[0] = 1.0
    moose.connect(pg, 'outputOut', sg, 'Vm')    
    return {
        'model': model_container,
        'iaf': iaf,
        'synchan': syn,
        'spikegen': sg,
        'pulsegen': pg}

if __name__ == '__main__':
    modelpath = '/lif_demo'
    datapath = '/data'
    simtime = 1.0
    setup = setupmodel(modelpath, 1.0, 1.0, 0.1)
    data_container = moose.Neutral(datapath)
    vm_table = moose.Table('%s/vm' % (data_container.path))
    moose.connect(vm_table, 'requestData', setup['iaf'], 'get_Vm')
    spike_table = moose.Table('%s/spike' % (data_container.path))
    moose.connect(spike_table, 'requestData', setup['spikegen'], 'get_hasFired')
    # moose.connect(setup['iaf'], 'VmOut', spike_table, 'spike')
    pulse_table = moose.Table('%s/pulse' % (data_container.path))
    moose.connect(pulse_table, 'requestData', setup['pulsegen'], 'get_output')
    gsyn_table = moose.Table('%s/gk' % (datapath))
    moose.connect(gsyn_table, 'requestData', setup['synchan'], 'get_Gk')
    moose.setClock(0, 1e-4)
    moose.setClock(1, 1e-4)
    moose.setClock(2, 1e-4)
    moose.setClock(3, 1e-4)
    moose.useClock(0, '%s,%s' % (setup['pulsegen'].path, setup['spikegen'].path), 'process')
    moose.useClock(1, setup['synchan'].path, 'process')
    moose.useClock(2, setup['iaf'].path, 'process')
    moose.useClock(3, '%s/##' % (datapath), 'process')
    moose.start(simtime)
    t = np.linspace(0, simtime, len(pulse_table.vec))
    plt.plot(t, pulse_table.vec * -1, 'g')
    print 'Spike table', spike_table.vec
    plt.plot(spike_table.vec, 'rx')
    print np.nonzero(np.array(spike_table.vec > 0))[0]
    # plt.plot(t, vm_table.vec * 1e3, 'b')
    print 'Vm table', vm_table.vec
    plt.plot(t, gsyn_table.vec, 'c')
    plt.show()

# 
# lif.py ends here
