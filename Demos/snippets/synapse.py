# synapse.py --- 
# 
# Filename: synapse.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat Jun 29 11:32:29 2013 (+0530)
# Version: 
# Last-Updated: Fri Jul  5 19:26:11 2013 (+0530)
#           By: subha
#     Update #: 113
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

def many_ematrix_to_one_element():
    """This is an example of event messages from multiple SpikeGen objects
    into a synchan.

    Create a SynChan element with 2 elements in synapse field.

    Create 5 SpikeGen elements.

    Connect alternet SpikeGen elements to synapse[0] and synapse[1]

    ... This is a minimal example. In real simulations the SpikeGens
    will be embedded in compartments representing axon terminals and
    the SynChans will be embedded in somatic/dendritic compartments.

    """
    model = moose.Neutral('/model')
    # data = moose.Neutral('/data')
    synchan = moose.SynChan('/model/synchan', 2)
    num_spikegen = 5
    spikegens = []
    tab = moose.Table('tab')
    tab.vec = [1, 2, 3]    
    print '%%%%'
    c = tab.y[1]
    print '%*%*%*'
    for ii in range(num_spikegen):
        spike = moose.ematrix('/model/spikegen_%d' % (ii), dtype='SpikeGen', dims=2)
        print 'SpikeId:', spike.value, spike.path,
        synchan.synapse.num += 1
        msg = moose.connect(spike, 'event', synchan.synapse[ii], 'addSpike')
        print '->', moose.element(msg).adjacent[spike[0]].path
        synchan.synapse[ii].delay = ii  * 1e-3
        synchan.synapse[ii].weight = (ii+1) * 0.1
    
if __name__ == '__main__':
    many_ematrix_to_one_element()

# 
# synapse.py ends here
