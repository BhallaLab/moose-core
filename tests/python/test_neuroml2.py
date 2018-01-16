# -*- coding: utf-8 -*-
# test_neurom2.py, modified from run_cell.py
# Description:
# Author:
# Maintainer: P Gleeson, Dilawar Singh
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

from __future__ import absolute_import, print_function, division

import moose
import moose.utils as mu
import sys
import numpy as np
    
def run(nogui):
    filename = 'test_files/passiveCell.nml'
    mu.info('Loading: %s'%filename)
    nml = moose.mooseReadNML2( filename )
    assert nml, "Expecting NML2 object"
    
    msoma = nml.getComp(nml.doc.networks[0].populations[0].id,0,0)
    mu.debug(msoma)
    
    data = moose.Neutral('/data')
    
    pg = nml.getInput('pulseGen1')
    
    inj = moose.Table('%s/pulse' % (data.path))
    moose.connect(inj, 'requestOut', pg, 'getOutputValue')
    
    
    vm = moose.Table('%s/Vm' % (data.path))
    moose.connect(vm, 'requestOut', msoma, 'getVm')
    
    simdt = 1e-6
    plotdt = 1e-4
    simtime = 150e-3
    
    if (1):
        #moose.showmsg( '/clock' )
        for i in range(8):
            moose.setClock( i, simdt )
        moose.setClock( 8, plotdt )
        moose.reinit()
    else:
        utils.resetSim([model.path, data.path], simdt, plotdt, simmethod='ee')
        moose.showmsg( '/clock' )
        
    moose.start(simtime)
    
    print("Finished simulation!")
    
    t = np.linspace(0, simtime, len(vm.vector))
    
    if not nogui:
        import matplotlib.pyplot as plt
        plt.subplot(211)
        plt.plot(t, vm.vector * 1e3, label='Vm (mV)')
        plt.legend()
        plt.title('Vm')
        plt.subplot(212)
        plt.title('Input')
        plt.plot(t, inj.vector * 1e9, label='injected (nA)')
        #plt.plot(t, gK.vector * 1e6, label='K')
        #plt.plot(t, gNa.vector * 1e6, label='Na')
        plt.legend()
        plt.show()
        plt.close()

    
if __name__ == '__main__':
    nogui = '-nogui' in sys.argv
    run(nogui)
