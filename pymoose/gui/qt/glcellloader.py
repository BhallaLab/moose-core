# glcellloader.py --- 
# 
# Filename: loadcell.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Thu Feb  4 20:34:34 2010 (+0530)
# Version: 
# Last-Updated: Sat Jun 26 15:50:44 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 107
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

import sys
import os

import moose

SIMDT = 1e-5
GLDT = 1e-2
RUNTIME = 1000e-3

morphs_dir = '../../../DEMOS/gl-demo/morphologies/'
models = {'CA1':'ca1passive.p',
	  'Mitral': 'mit.p',
	  'Purkinje1': 'psmall.p',
	  'Purkinje2': 'Purk2M9s.p',
	  'Purkinje3': 'Purkinje4M9.p'
	  }

CONTEXT = moose.PyMooseBase.getContext()

class GLCellLoader(object):
    def __init__(self, cell_type, host='localhost', port='9999'):
	'''Cell loader for glcell using glclient'''
	filepath = morphs_dir + models[cell_type]

        # Load the channel definitions from bulbchan.g
        CONTEXT.loadG('../../../DEMOS/mitral-ee/bulbchan.g')
        cwe = CONTEXT.getCwe()
        CONTEXT.setCwe('/library')
        CONTEXT.runG('make_LCa3_mit_usb')
        CONTEXT.runG('make_Na_rat_smsnn')
        CONTEXT.runG('make_Na2_rat_smsnn')
        CONTEXT.runG('make_KA_bsg_yka')
        CONTEXT.runG('make_KM_bsg_yka')
        CONTEXT.runG('make_K_mit_usb')
        CONTEXT.runG('make_K2_mit_usb')
        # CONTEXT.runG('make_K_slow_usb')
        CONTEXT.runG('make_Na_mit_usb')
        CONTEXT.runG('make_Na2_mit_usb')
        # CONTEXT.runG('make_Ca_mit_conc')
        # CONTEXT.runG('make_Kca_mit_usb')
        print 'created channels'
        CONTEXT.setCwe(cwe)
	CONTEXT.readCell(filepath, cell_type)
	self.cell = moose.Cell(cell_type)
	self.glServer = moose.GLcell('gl_' + cell_type)
	self.glServer.vizpath = self.cell.path
	self.glServer.port = port
	self.glServer.host = host
	self.glServer.attribute = 'Vm'
	self.glServer.threshold = 1
	self.glServer.sync = 'on'
	self.glServer.vscale = 1.0
	self.glServer.bgcolor = '050050050'
	self.glServer.highvalue = 0.05
	self.glServer.loader = -0.1
	if cell_type == 'Mitral':
            self.glServer.vscale = 10.0
	# ** Assuming every cell has a top-level compartment called
	# ** soma
	self.pulsegen = moose.PulseGen('pg_' + cell_type)
	self.pulsegen.firstDelay = 5e-3
	self.pulsegen.firstWidth = 50e-3
	self.pulsegen.firstLevel = 1e-9
	self.pulsegen.connect('outputSrc', moose.Compartment(self.cell.path + '/soma'), 'injectMsg')


if __name__ == '__main__':
    if len(sys.argv) > 1:
	loader = GLCellLoader(sys.argv[1])
    else:
	loader = GLCellLoader('Purkinje3')
    print 'loaded morphology file'
    CONTEXT.setClock(0, SIMDT)
    CONTEXT.setClock(1, SIMDT)
    CONTEXT.setClock(2, SIMDT)
    CONTEXT.setClock(4, GLDT)
    CONTEXT.useClock(4, '/#[TYPE=GLcell]')
    print 'Before reset'
    CONTEXT.reset()
    print 'After reset'
    CONTEXT.step(RUNTIME)

# 
# glcellloader.py ends here
