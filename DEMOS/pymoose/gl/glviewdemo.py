# glviewdemo.py --- 
# 
# Filename: glviewdemo.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Fri Feb  5 16:45:13 2010 (+0530)
# Version: 
# Last-Updated: Tue Jul 13 14:45:36 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 14
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
import os
import time
import sys
import subprocess
import moose

from glclient import GLClient

SIMDT = 1e-5
GLDT = 1e-2
RUNTIME = 100e-3

morphs_dir = 'morphologies'
models = {'CA1':'ca1passive.p',
	  'Mitral': 'mit.p',
	  'Purkinje1': 'psmall.p',
	  'Purkinje2': 'Purk2M9s.p',
	  'Purkinje3': 'Purkinje4M9.p'
	  }

CONTEXT = moose.PyMooseBase.getContext()


class GLViewDemo(object):
    '''Demo showing a cell in glview mode.
    
    By default it loads a Mitral Cell model and shows the compartments
    as cubes placed in 3D. The startup is always a profile view and if
    you rotate it downwards to have the topview (axon being the
    bottom), you will see nice changes in color and size of the cubes
    indicating the propagation of spikes.
    '''
    def __init__(self, port=9999, colormap=os.path.join('colormaps', 'rainbow2'), celltype='Mitral'):
	self.client = GLClient(port=str(port), colormap=colormap, mode='v')
	time.sleep(3) # Without a little delay the client gives bind error
        # create the channels for Mitral cell.        
	self.server = subprocess.Popen(['python', 'glviewloader.py', celltype])

if __name__ == '__main__':
    '''main mathod for running glcell demo. There are four models associated with this demo:
       They are the following with corresponding command line option:

       mitral cell          : Mitral
       purkinje cell (small): Purkinje1
       purkinje cell        : Purkinje2
       purkinje cell        : Purkinje3


       By default it runs the Mitral cell model.
       The models are not exact as all the ion channels may not be present.'''
    celltype = 'Mitral'
    if len(sys.argv) > 1:
	celltype = sys.argv[1]
    
    demo = GLViewDemo(celltype=celltype)
    
	



# 
# glviewdemo.py ends here
