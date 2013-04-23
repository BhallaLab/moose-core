# test_converter.py --- 
# 
# Filename: test_converter.py
# Description: 
# Author: 
# Maintainer: 
# Created: Tue Apr 23 18:51:58 2013 (+0530)
# Version: 
# Last-Updated: Tue Apr 23 19:06:45 2013 (+0530)
#           By: subha
#     Update #: 37
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

import uuid
import unittest
import moose
import converter
import neuroml

class TestConvertMorphology(unittest.TestCase):
    def setUp(self):
        self.test_id = uuid.uuid4()
        self.model_container = moose.Neutral('test%s' % (self.test_id))
        self.neuron = moose.Neuron('%s/cell' % (self.model_container.path))
        self.soma = moose.Compartment('%s/soma' % (self.neuron.path))
        self.soma.diameter = 20e-6
        self.soma.length = 0.0
        parent = self.soma
        comps = []
        for ii in range(100):
            comp = moose.Compartment('%s/comp_%d' % (self.neuron.path, ii))
            comp.diameter = 10e-6
            comp.length = 100e-6
            moose.connect(parent, 'raxial', comp, 'axial')
            comps.append(comp)
            parent = comp
    
    def test_convert_morphology(self):
        morph = converter.convert_morphology(self.neuron, positions='auto')
        cell = neuroml.Cell()
        cell.name = self.neuron.name
        cell.id = cell.name
        cell.morphology = morph
        doc = neuroml.NeuroMLDocument()
        doc.cells.append(cell)
        doc.id = 'TestNeuroMLDocument'
        fname = './tmp/test_morphology_conversion.xml'
        neuroml.writers.NeuroMLWriter.write(doc, fname)

if __name__ == '__main__':
    unittest.main()
        


# 
# test_converter.py ends here
