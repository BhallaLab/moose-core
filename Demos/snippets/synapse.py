# synapse.py --- 
# 
# Filename: synapse.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat Jun 29 11:32:29 2013 (+0530)
# Version: 
# Last-Updated: Sat Jun 29 11:59:46 2013 (+0530)
#           By: subha
#     Update #: 50
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
    synchan = moose.SynChan('/model/synchan')
    synchan.synapse.num = 2
    num_spikegen = 5
    spikegens = []
    for ii in range(num_spikesource):
        spike = moose.SpikeGen('/model/spikegen_%d' % (ii))
        msg = moose.connect(spike, 'event', synchan.synapse, 'addSpike', 'Sparse')
        # This will create sparse_matrix[0][0] = synapse_index.
        # if ii is even, synapse_index=0, 1 otherwise
        msg.setEntry(0, 0, ii%2)
    


# 
# synapse.py ends here
