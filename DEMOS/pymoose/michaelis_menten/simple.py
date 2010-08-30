# simple.py --- 
# 
# Filename: simple.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Mon Aug 30 11:54:20 2010 (+0530)
# Version: 
# Last-Updated: Mon Aug 30 13:22:41 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 85
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# A simple demonstration of Michaelis-Menten reaction with one substrate
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
import pylab
import numpy as np

import moose

class MichaelisMenten(object):
    """A very simple Michaelis-Menten reaction"""
    def __init__(self):
        self.context = moose.PyMooseBase.getContext()
        self.kin_compt = moose.KinCompt('mm')
        self.enz_mol = moose.Molecule('enzMol', self.kin_compt)
        self.enzyme = moose.Enzyme('enzyme', self.enz_mol)
        self.substrate = moose.Molecule('substrate', self.kin_compt)
        self.product = moose.Molecule('product', self.kin_compt)
        self.enzyme.connect('sub', self.substrate, 'reac')
        self.enzyme.connect('prd', self.product, 'prd')
        self.enz_mol.connect('reac', self.enzyme, 'enz')
        self.kin_compt.volume = 1.0
        self.substrate.concInit = 1.0
        self.product.concInit = 0.0
        self.enz_mol.concInit = 1.0
        self.enzyme.k1 = 0.1
        self.enzyme.k2 = 0.4
        self.enzyme.k3 = 0.1
        self.data = moose.Neutral('/data')
        self.sub_conc_table = moose.Table('sub_conc_table', self.data)
        self.prd_conc_table = moose.Table('prd_conc_table', self.data)
        self.sub_conc_table.stepMode = 3
        self.prd_conc_table.stepMode = 3
        self.sub_conc_table.connect('inputRequest', self.substrate, 'conc')
        self.prd_conc_table.connect('inputRequest', self.product, 'conc')
        self.dt = 1.0
        self.simtime = 1000.0
    
    def run(self):
        self.context.setClock(0, self.dt)
        self.context.setClock(1, self.dt)
        self.context.setClock(2, self.dt)
        self.context.setClock(3, self.dt)
        print 'Before reset'
        self.context.reset()
        print 'After reset'
        self.context.step(self.simtime)
        pylab.plot(np.array(self.prd_conc_table))
        pylab.plot(np.array(self.sub_conc_table))
        pylab.show()

    def set_Km_slot(self, Km):
        """For use by Qt classes to set Km"""
        self.enzyme.Km = Km

    def set_k2_slot(self, k2):
        """For use by Qt classes to set k2"""
        self.enzyme.k2 = k2

    def set_substrate_conc_slot(self, conc):
        """For use by Qt classes to set initial substrate
        conecnetration."""
        self.substrate.concInit = conc

    def set_product_conc_slot(self, conc):
        """For use by Qt classes to set initial product
        concentration"""
        self.product.concInit = conc

    def set_enzyme_conc_slot(self, conc):
        """For use by Qt classes to set initial enzyme
        concentration."""
        self.enz_mol.concInit = conc

    def set_simtime_slot(self, simtime):
        """For use by Qt classes to set simulation run time."""
        self.simtime = simtime

    def set_dt_slot(self, dt):
        """For use by Qt classes to set integration time step"""
        self.dt = dt

        

if __name__ == '__main__':
    mm = MichaelisMenten()
    mm.run(10000)
    
        


# 
# simple.py ends here
