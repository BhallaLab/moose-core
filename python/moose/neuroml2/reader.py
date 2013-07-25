# reader.py --- 
# 
# Filename: reader.py
# Description: 
# Author: 
# Maintainer: 
# Created: Wed Jul 24 15:55:54 2013 (+0530)
# Version: 
# Last-Updated: Wed Jul 24 16:48:17 2013 (+0530)
#           By: subha
#     Update #: 26
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
"""Implementation of reader for NeuroML 2 models"""

import neuroml
import neuroml.loaders as loaders

class NML2Reader(object):
    """Reads NeuroML2 and creates MOOSE model"""
    def __init__(self):
        self.doc = None
        self.filename = None
        
    def read(self, filename):
        self.doc = loaders.NeuroMLLoader.load(filename)
        self.filename = filename
        print 'Loaded file from', filename

    def createCellPrototype(self, index):
        """To be completed - create the morphology, channels in prototype"""
        cell = self.doc.cells[index]
        return cell
    
        



# 
# reader.py ends here
