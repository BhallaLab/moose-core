# nsdf.py --- 
# 
# Filename: nsdf.py
# Description: 
# Author: subha
# Maintainer: 
# Created: Fri Jun 26 12:23:07 2015 (-0400)
# Version: 
# Last-Updated: 
#           By: 
#     Update #: 0
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
"""This script is for demonstrating the use of NSDFWriter class to
dump data in NSDF format"""

import numpy as np

import moose

def setup_model():
    """Setup a dummy model with a pulsegen and a spikegen detecting the
    leading edges of the pulses. We record the pulse output as Uniform
    data and leading edge time as Event data."""
    model = moose.Neutral('/model')
    pulse = moose.PulseGen('/model/pulse')
    pulse.level[0] = 1.0
    pulse.delay[0] = 0.5
    pulse.width[0] = 1.0
    t_lead = moose.SpikeGen('/model/t_lead')
    t_lead.threshold = 0.5
    moose.connect(pulse, 'output', t_lead,'Vm');
    nsdf = moose.NSDFWriter('/model/writer')
    nsdf.filename = 'nsdf_demo.h5'
    nsdf.mode = 2 #overwrite existing file
    moose.connect(nsdf, 'requestOut', pulse, 'getOutputValue')
    print 'event input', nsdf.eventInput, nsdf.eventInput.num
    print nsdf

    nsdf.eventInput.num = 1
    ei = nsdf.eventInput[0]
    print ei.path
    moose.connect(t_lead, 'spikeOut', nsdf.eventInput[0], 'input')
    tab = moose.Table('spiketab')
    tab.threshold = t_lead.threshold
    moose.connect(pulse, 'output', tab, 'spike')
    moose.reinit()
    moose.start(100)
    np.savetxt('nsdf.txt', tab.vector)

if __name__ == '__main__':
    setup_model()


# 
# nsdf.py ends here
