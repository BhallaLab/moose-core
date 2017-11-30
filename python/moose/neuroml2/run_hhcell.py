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
    plt.show()
    plt.close()

    
    