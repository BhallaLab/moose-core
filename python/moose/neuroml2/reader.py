# reader.py --- 
# 
# Filename: reader.py
# Description: 
# Author: 
# Maintainer: 
# Created: Wed Jul 24 15:55:54 2013 (+0530)
# Version: 
# Last-Updated: Fri Jul 26 17:06:35 2013 (+0530)
#           By: subha
#     Update #: 383
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

import numpy as np
import moose
import generated_neuroml as nml
from units import SI

def sarea(comp):
    """Calculate the surface area of compartment from length and
    diameter"""
    if comp.length > 0:
        return comp.length * comp.diameter * np.pi
    else:
        return comp.diameter * comp.diameter * np.pi

def xarea(comp):
    """Calculate the cross sectional area from diameter of the
    compartment. How to do it for spherical compartment?"""
    return comp.diameter * comp.diameter * np.pi / 4.0

def setRa(comp, resistivity):
    """Calculate total raxial from specific value `resistivity`"""
    if comp.length > 0:
        comp.Ra = resistivity * comp.length / xarea(comp)
    else:
        comp.Ra = resistivity * 8.0 / (comp.diameter * np.pi)

def setRm(comp, resistivity):
    """Set membrane resistance"""
    comp.Rm = resistivity / sarea(comp)
    

def getSegments(nmlcell, component, sg_to_segments):
    """Get the list of segments the `component` is applied to"""
    sg = segment.segmentGroup
    seg = segment.segment
    if sg is None:
        if seg:
            segments = [seg]
        else:
            segments = nmlcell.morphology.segment
    else:
        segments = sg_to_segments[sg]
    return segments
    

class NML2Reader(object):
    """Reads NeuroML2 and creates MOOSE model"""
    def __init__(self):
        self.lunit = 1e-6 # micron is the default length unit
        self.doc = None
        self.filename = None        
        self.nml_to_moose = {} # NeuroML object to MOOSE object
        self.moose_to_nml = {} # Moose object to NeuroML object
        self.proto_cells = {} # map id to prototype cell in moose
        self.proto_chans = {} # map id to prototype channels in moose
        self.lib = moose.Neutral('/library')
        self._cell_to_sg = {} # nml cell to dict - the dict maps segment groups to segments
        
    def read(self, filename):
        self.doc = nml.parse(filename)
        self.filename = filename

    def createCellPrototype(self, index, symmetric=False):
        """To be completed - create the morphology, channels in prototype"""
        cell = self.doc.cell[index]
        nrn = moose.Neuron('%s/%s' % (self.lib.path, cell.id))
        self.proto_cells[cell.id] = nrn
        self.nml_to_moose[cell] = nrn
        self.moose_to_nml[nrn] = cell
        self.createMorphology(cell, nrn, symmetric=symmetric)
        self.importBiophysics(cell, nrn)
        return cell, nrn
    
    def createMorphology(self, nmlcell, moosecell, symmetric=False):
        """Create the MOOSE compartmental morphology in `moosecell` using the
        segments in NeuroML2 cell `nmlcell`. Create symmetric
        compartments if `symmetric` is True.

        """
        morphology = nmlcell.morphology
        segments = morphology.segment
        id_to_segment = dict([(seg.id, seg) for seg in segments])    
        if symmetric:
            compclass = moose.SymCompartment
        else:
            compclass = moose.Compartment
        # segment names are used as compartment names - assuming
        # naming convention does not clash with that in MOOSE
        cellpath = moosecell.path
        id_to_comp = {}
        for seg in segments:
            if seg.name is not None:
                id_to_comp[seg.id] = compclass('%s/%s' % (cellpath, seg.name))
            else:
                id_to_comp[seg.id] = compclass('%s/comp_%s' % (cellpath, seg.id))
        # Now assign the positions and connect up axial resistance
        if not symmetric:
            src, dst = 'axial', 'raxial'
        else:
            src, dst = 'proximal', 'distal'
        for segid, comp in id_to_comp.items():
            segment = id_to_segment[segid]
            try:
                parent = id_to_segment[str(segment.parent.segment)]
            except AttributeError:
                parent = None
            self.moose_to_nml[comp] = segment
            self.nml_to_moose[segment] = comp            
            p0 = segment.proximal            
            if p0 is None:
                if parent:
                    p0 = parent.distal
                else:
                    raise Exception('No proximal point and no parent segment for segment: name=%s, id=%s' % (segment.name, segment.id))
            comp.x0, comp.y0, comp.z0 = map(lambda x: x * self.lunit, map(float, (p0.x, p0.y, p0.z)))            
            p1 = segment.distal
            comp.x, comp.y, comp.z = map(lambda x: x * self.lunit, map(float, (p1.x, p1.y, p1.z)))
            comp.length = np.sqrt((comp.x - comp.x0)**2
                                  + (comp.y - comp.y0)**2
                                  + (comp.z - comp.z0)**2)
            # This can pose problem with moose where both ends of
            # compartment have same diameter. We are averaging the two
            # - may be splitting the compartment into two is better?
            comp.diameter = (float(p0.diameter)+float(p1.diameter)) * self.lunit / 2
            if parent:
                pcomp = id_to_comp[parent.id]
                moose.connect(comp, src, pcomp, dst)
        sg_to_segments = {}        
        for sg in morphology.segmentGroup:
            sg_to_segments[sg.id] = [id_to_segment[str(m.segment)] for m in sg.member]
        self._cell_to_sg[nmlcell] = sg_to_segments
        return id_to_comp, id_to_segment, sg_to_segments
            
    def importBiophysics(self, nmlcell, moosecell):
        """Create the biophysical components in moose Neuron `moosecell`
        according to NeuroML2 cell `nmlcell`."""
        bp = nmlcell.biophysicalProperties
        self.importMembraneProperties(nmlcell, moosecell)
        # self.createIntracellularProperties(nmlcell, moosecell)

    def importMembraneProperties(self, nmlcell, moosecell):
        """Create the membrane properties from nmlcell in moosecell"""
        mp = nmlcell.biophysicalProperties.membraneProperties
        self.importCapacitances(nmlcell, moosecell, mp.specificCapacitance)

    def importCapacitances(self, nmlcell, moosecell, specificCapacitances):
        sg_to_segments = self._cell_to_sg[nmlcell]
        for specific_cm in specificCapacitances:
            cm = SI(specific_cm.value)
            for seg in sg_to_segments[specific_cm.segmentGroup]:
                comp = self.nml_to_moose[seg]
                comp.Cm = np.pi * sarea(comp) 

    def importIntracellularProperties(self, nmlcell, moosecell, membraneProperties):
        self.importAxialResistance(nmlcell, membraneProperties.intraCellularProperties)
        # TODO CaConc

    def importAxialResistance(self, nmlcell, intraCellularProperties):
        sg_to_segments = self._cell_to_sg[nmlcell]
        for r in intraCellularProperties.resistivity:
            segments = get_segments(nmlcell, r, sg_to_segments)
            for seg in segments:
                comp = self.nml_to_moose[seg]
                setRa(comp, SI(r.value))                    

    def importChannels(self, nmlcell, moosecell, membraneProperties):
        sg_to_segments = self._cell_to_sg[nmlcell]
        for chdens in membraneProperties.channelDensity:
            segments = get_segments(nmlcell, chdens, sg_to_segments)
            condDensity = SI(chdens.condDensity)
            if self.id_to_ionChannel[chdens.ionChannel].type == 'ionChannelPassive':
                for seg in segments:
                    self.setRm(self.nml_to_moose[seg], condDensity)
            else:
                for seg in segments:
                    self.copyChannel(chdens, self.nml_to_moose[seg], condDensity)
                
    def copyChannel(self, chdens, comp, condDensity):
        """Copy moose prototype for `chdens` condutcance density to `comp`
        compartment.

        """
        proto_chan = self.proto_chans[chdens.ionChannel]
        chid = moose.copy(proto_chan, comp)
        chan = moose.element(chid)
        chan.Gbar = sarea(comp) * condDensity
        moose.connect(chan, 'channel', comp, 'channel')



# 
# reader.py ends here
