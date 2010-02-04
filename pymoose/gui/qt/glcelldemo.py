# glcelldemo.py --- 
# 
# Filename: glcelldemo.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Thu Feb  4 19:58:31 2010 (+0530)
# Version: 
# Last-Updated: Thu Feb  4 22:12:48 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 34
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

import time
import sys
import subprocess
from glclient import GLClient

class GlCellDemo(object):
    def __init__(self, port=9999, colormap='../../../gl/colormaps/rainbow2', celltype='Purkinje3'):
	self.client = GLClient(port=str(port), colormap=colormap)
	time.sleep(3) # Without a little delay the client gives bind error
	self.server = subprocess.Popen(['python', 'loadcell.py', celltype])
if __name__ == '__main__':
    celltype = 'Purkinje3'
    if len(sys.argv) > 1:
	celltype = sys.argv[1]
    
    demo = GlCellDemo(celltype=celltype)
    
	


# 
# glcelldemo.py ends here
