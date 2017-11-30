# -*- coding: utf-8 -*-
# run_hhcell.py ---
#
# Filename: run_hhcell.py
# Description:
# Author:
# Maintainer: P Gleeson
# Version:
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

import moose
from reader import NML2Reader
import numpy as np
import matplotlib.pyplot as plt


def test_channel_gates():
    """Creates prototype channels under `/library` and plots the time
    constants (tau) and activation (minf, hinf, ninf) parameters for the
    channel gates.

    """
    lib = moose.Neutral('/library')
    m = moose.element('/library[0]/naChan[0]/gateX')
    h = moose.element('/library[0]/naChan[0]/gateY')
    n = moose.element('/library[0]/kChan[0]/gateX')
    v = np.linspace(n.min,n.max, n.divs+1)
    
    plt.subplot(221)
    plt.plot(v, 1/m.tableB, label='tau_m')
    plt.plot(v, 1/h.tableB, label='tau_h')
    plt.plot(v, 1/n.tableB, label='tau_n')
    plt.legend()
    
    plt.subplot(222)
    plt.plot(v, m.tableA/m.tableB, label='m_inf')
    plt.plot(v, h.tableA/h.tableB, label='h_inf')
    plt.plot(v, n.tableA/n.tableB, label='n_inf')
    plt.legend()
    
    plt.subplot(223)
    plt.plot(v, m.tableA, label='mA(alpha)')
    plt.plot(v, h.tableA, label='hA(alpha)')
    plt.plot(v, n.tableA, label='nA(alpha)')
    plt.legend()
    plt.subplot(224)
    
    plt.plot(v, m.tableB, label='mB')
    plt.plot(v, m.tableB-m.tableA, label='mB-A(beta)')
    
    plt.plot(v, h.tableB, label='hB')
    plt.plot(v, h.tableB-h.tableA, label='hB-A(beta)')
    
    plt.plot(v, n.tableB, label='nB')
    plt.plot(v, n.tableB-n.tableA, label='nB-nA(beta)')
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
    
    reader = NML2Reader(verbose=True)

    lib = moose.Neutral('/library')
    filename = 'test_files/NML2_SingleCompHHCell.nml'
    print('Loading: %s'%filename)
    reader.read(filename)
    
    print(reader.doc.id)
    cell = reader.doc.cell[0]
    cell_id = cell.id
    soma = cell.morphology.segment[0]
    print(cell_id)
    print(soma.id)
    print(reader.proto_cells[cell_id])
    print(reader.nml_to_moose)
    msoma = reader.nml_to_moose[soma]
    print(msoma)
    
    
    data = moose.Neutral('/data')
    pg = moose.PulseGen('%s/pg' % (lib.path))
    pg.firstDelay = 100e-3
    pg.firstWidth = 100e-3
    pg.firstLevel = 0.08e-9
    pg.secondDelay = 1e9
    moose.connect(pg, 'output', msoma, 'injectMsg')
    inj = moose.Table('%s/pulse' % (data.path))
    moose.connect(inj, 'requestOut', pg, 'getOutputValue')
    
    
    vm = moose.Table('%s/Vm' % (data.path))
    moose.connect(vm, 'requestOut', msoma, 'getVm')
    
    simdt = 1e-6
    plotdt = 1e-4
    simtime = 300e-3
    if (1):
        moose.showmsg( '/clock' )
        for i in range(8):
            moose.setClock( i, simdt )
        moose.setClock( 8, plotdt )
        moose.reinit()
    else:
        utils.resetSim([model.path, data.path], simdt, plotdt, simmethod='ee')
        moose.showmsg( '/clock' )
    moose.start(simtime)
    
    t = np.linspace(0, simtime, len(vm.vector))
    vfile = open('moose_v_hh.dat','w')
    
    for i in range(len(t)):
        vfile.write('%s\t%s\n'%(t[i],vm.vector[i]))
    vfile.close()
    plt.subplot(211)
    plt.plot(t, vm.vector * 1e3, label='Vm (mV)')
    plt.plot(t, inj.vector * 1e9, label='injected (nA)')
    plt.legend()
    plt.title('Vm')
    plt.subplot(212)
    plt.title('Conductance (uS)')
    #plt.plot(t, gK.vector * 1e6, label='K')
    #plt.plot(t, gNa.vector * 1e6, label='Na')
    plt.legend()
    plt.figure()
    test_channel_gates()
    plt.show()
    plt.close()

    
    