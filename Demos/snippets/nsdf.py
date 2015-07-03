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
from datetime import datetime
import getpass

import moose

def setup_model():
    """Setup a dummy model with a pulsegen and a spikegen detecting the
    leading edges of the pulses. We record the pulse output as Uniform
    data and leading edge time as Event data."""
    model = moose.Neutral('/model')
    pulse = moose.PulseGen('/model/pulse')
    pulse.level[0] = 1.0
    pulse.delay[0] = 10
    pulse.width[0] = 20
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
    clock = moose.element('/clock')
    for ii in range(32):
        moose.setClock(ii, 1)
    moose.useClock(30, nsdf.path, 'process')
    moose.connect(pulse, 'output', tab, 'spike')
    print datetime.now().isoformat()
    moose.reinit()
    moose.start(100)
    print datetime.now().isoformat()
    np.savetxt('nsdf.txt', tab.vector)
    ###################################
    # Set the environment attributes
    ###################################
    nsdf.stringAttr['title'] = 'NSDF writing demo for moose'
    nsdf.stringAttr['description'] = '''An example of writing data to NSDF file from MOOSE simulation. In
this simulation we generate square pules from a PulseGen object and
use a SpikeGen to detect the threshold crossing events of rising
edges. We store the pulsegen output as Uniform data and the threshold
crossing times as Event data. '''    
    nsdf.stringAttr['creator'] = getpass.getuser()
    nsdf.stringVecAttr['software'] = ['python2.7', 'moose3' ]
    nsdf.stringVecAttr['method'] = ['']
    nsdf.stringAttr['rights'] = ''
    nsdf.stringAttr['license'] = 'CC-BY-NC'

    


if __name__ == '__main__':
    setup_model()


# 
# nsdf.py ends here
