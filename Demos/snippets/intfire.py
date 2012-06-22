# intfire.py --- 
# 
# Filename: intfire.py
# Description: 
# Author: 
# Maintainer: 
# Created: Thu Jun 21 16:40:25 2012 (+0530)
# Version: 
# Last-Updated: Thu Jun 21 16:46:56 2012 (+0530)
#           By: subha
#     Update #: 20
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Code snippet to show some operations on IntFire.
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

def connect_two_intfires():
    """Connect two IntFire neurons so that spike events in one gets
    transmitted to synapse of the other."""
    if1 = moose.IntFire('if1')
    if2 = moose.IntFire('if2')
    syn1 = moose.element(if1.path + '/synapse')
    # Connect the spike message of if2 to the first synapse on if1
    moose.connect(if2, 'spike', syn1, 'addSpike')

def connect_spikegen():
    """Connect a SpikeGen object to an IntFire neuron such that spike
    events in spikegen get transmitted to the synapse of the IntFire
    neuron."""
    if3 = moose.IntFire('if3')
    sg = moose.SpikeGen('sg')
    syn = moose.element(if3.path + '/synapse')
    moose.connect(sg, 'event', if3, 'addSpike')



# 
# intfire.py ends here
