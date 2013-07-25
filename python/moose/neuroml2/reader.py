# reader.py --- 
# 
# Filename: reader.py
# Description: 
# Author: 
# Maintainer: 
# Created: Wed Jul 24 15:55:54 2013 (+0530)
# Version: 
# Last-Updated: Thu Jul 25 08:35:48 2013 (+0530)
#           By: subha
#     Update #: 110
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

import moose
import neuroml
import neuroml.loaders as loaders

class NML2Reader(object):
    """Reads NeuroML2 and creates MOOSE model"""
    def __init__(self):
        self.doc = None
        self.filename = None        
        self.nml_to_moose = {} # NeuroML object to MOOSE object
        self.moose_to_nml = {} # Moose object to NeuroML object
        self.proto_cells = {}
        self.proto_chans = {}
        self.lib = moose.Neutral('/library')
        
    def read(self, filename):
        self.doc = loaders.NeuroMLLoader.load(filename)
        self.filename = filename
        print 'Loaded file from', filename

    def createCellPrototype(self, index, symmetric=False):
        """To be completed - create the morphology, channels in prototype"""
        cell = self.doc.cells[index]
        print dir(cell)
        nrn = moose.Neuron('%s/%s' % (self.lib.path, cell.id))
        self.proto_cells[cell.id] = nrn
        self.nml_to_moose[cell] = nrn
        self.moose_to_nml[nrn] = cell
        self.createMorphology(cell, nrn, symmetric=symmetric)
        return cell, nrn
    
    def createMorphology(self, nmlcell, moosecell, symmetric=False):
        """Create the MOOSE compartmental morphology in `moosecell` using the
        segments in NeuroML2 cell `nmlcell`. Create symmetric
        compartments if `symmetric` is True.

        """
        id_to_segment = dict([(seg.id, seg) for seg in nmlcell.morphology.segments])        
        if symmetric:
            compclass = moose.SymCompartment
        else:
            compclass = moose.Compartment
        # segment names are used as compartment names - assuming
        # naming convention does not clash with that in MOOSE
        cellpath = moosecell.path
        try:
            id_to_comp = dict([(seg.id, compclass('%s/%s' % (cellpath, seg.name)))
                               for seg in nmlcell.morphology.segments])
        except AttributeError: # name is not a mandatory attribute, id is
            id_to_comp = dict([(seg.id, compclass('%s/comp_%s' % (cellpath, seg.id)))
                               for seg in nmlcell.morphology.segments])
        # Now assign the positions and connect up axial resistance
        if not symmetric:
            src = 'axial'
            dst = 'raxial'
        else:
            src = 'proximal'
            dst = 'distal'
        for segid, comp in id_to_comp.items():
            segment = id_to_segment[segid]
            self.moose_to_nml[comp] = segment
            self.nml_to_moose[segment] = comp
            p0 = segment.proximal
            comp.x0, comp.y0, comp.z0 = apply(float, (p0.x, p0.y, p0.z))
            p1 = segment.distal
            comp.x, comp.y, comp.z = apply(float, (p1.x, p1.y, p1.z))
            try:
                parent = id_to_comp[segment.parent.segment]
                moose.connect(comp, src, parent, dst)
            except AttributeError:
                pass
            

# 
# reader.py ends here
