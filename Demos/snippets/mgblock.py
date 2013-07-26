# mgblock.py --- 
# 
# Filename: mgblock.py
# Description: 
# Author: 
# Maintainer: 
# Created: Wed Jul  3 09:36:06 2013 (+0530)
# Version: 
# Last-Updated: Fri Jul 26 15:33:18 2013 (+0530)
#           By: subha
#     Update #: 115
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

"""Demonstrates the use of MgBlock"""
import moose
from moose import utils
import pylab

simtime = 100e-3
simdt = 1e-6
plotdt = 1e-4

def test_mgblock():
    model = moose.Neutral('/model')
    data = moose.Neutral('/data')
    soma = moose.Compartment('/model/soma')
    soma.Em = -60e-3
    soma.Rm = 1e7
    soma.Cm = 1e-9

    ###################################################
    # This is where we create the synapse with MgBlock
    #--------------------------------------------------
    nmda = moose.SynChan('/model/soma/nmda')
    nmda.Gbar = 1e-9
    mgblock = moose.MgBlock('/model/soma/mgblock')
    mgblock.CMg = 2.0
    mgblock.KMg_A = 1/0.33
    mgblock.KMg_B = 1/60.0
    
    # MgBlock sits between original channel nmda and the
    # compartment. The origChannel receives the channel message from
    # the nmda SynChan.
    moose.connect(nmda, 'channelOut', mgblock, 'origChannel')
    moose.connect(mgblock, 'channel', soma, 'channel')    
    # This is for comparing with MgBlock
    nmda_noMg = moose.element(moose.copy(nmda, soma, 'nmda_noMg'))
    moose.connect(moose.element(nmda_noMg), 'channel', soma, 'channel')

    #########################################
    # The rest is for experiment setup
    spikegen = moose.SpikeGen('/model/spike')
    pulse = moose.PulseGen('/model/input')
    pulse.delay[0] = 10e-3
    pulse.level[0] = 1.0
    pulse.width[0] = 50e-3
    moose.connect(pulse, 'outputOut', spikegen, 'Vm')
    nmda.synapse.num = 1
    syn = moose.element(nmda.synapse.path)
    moose.connect(spikegen, 'event', syn, 'addSpike')
    nmda_noMg.synapse.num = 1
    moose.connect(spikegen, 'event', moose.element(nmda_noMg.synapse.path), 'addSpike')
    Gnmda = moose.Table('/data/Gnmda')
    moose.connect(Gnmda, 'requestData', mgblock, 'get_Gk')
    Gnmda_noMg = moose.Table('/data/Gnmda_noMg')
    moose.connect(Gnmda_noMg, 'requestData', nmda_noMg, 'get_Gk')
    Vm = moose.Table('/data/Vm')
    moose.connect(Vm, 'requestData', soma, 'get_Vm')
    utils.setDefaultDt(elecdt=simdt, plotdt2=plotdt)
    utils.assignDefaultTicks(modelRoot='/model', dataRoot='/data')
    moose.reinit()
    utils.stepRun(simtime, simtime/10)
    for ii in range(10):
        for n in moose.element('/clock/tick').neighbours['proc%d' % (ii)]:
            print ii, n.path
    t = pylab.linspace(0, simtime*1e3, len(Vm.vec))
    pylab.plot(t, Vm.vec*1e3, label='Vm (mV)')
    pylab.plot(t, Gnmda.vec * 1e9, label='Gnmda (nS)')
    pylab.plot(t, Gnmda_noMg.vec * 1e9, label='Gnmda no Mg (nS)')
    pylab.legend()
    data = pylab.vstack((t, Gnmda.vec, Gnmda_noMg.vec)).transpose()
    pylab.savetxt('mgblock.dat', data)
    pylab.show()
    

if __name__ == '__main__':
    test_mgblock()

# 
# mgblock.py ends here
