# rdesignerProtos.py --- 
# 
# Filename: rdesignerProtos.py
# Description: 
# Author: Subhasis Ray, Upi Bhalla
# Maintainer: 
# Created: Tue May  7 12:11:22 2013 (+0530)
# Version: 
# Last-Updated: Wed Dec 30 13:01:00 2015 (+0530)
#           By: Upi
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
import numpy as np
import moose
from moose import utils

EREST_ACT = -70e-3
per_ms = 1e3

def make_HH_Na(name = 'HH_Na', parent='/library', vmin=-110e-3, vmax=50e-3, vdivs=3000):
    """Create a Hodhkin-Huxley Na channel under `parent`.
    
    vmin, vmax, vdivs: voltage range and number of divisions for gate tables
    
    """
    na = moose.HHChannel('%s/%s' % (parent, name))
    na.Ek = 50e-3
    na.Xpower = 3
    na.Ypower = 1
    v = np.linspace(vmin, vmax, vdivs+1) - EREST_ACT
    m_alpha = per_ms * (25 - v * 1e3) / (10 * (np.exp((25 - v * 1e3) / 10) - 1))
    m_beta = per_ms * 4 * np.exp(- v * 1e3/ 18)
    m_gate = moose.element('%s/gateX' % (na.path))
    m_gate.min = vmin
    m_gate.max = vmax
    m_gate.divs = vdivs
    m_gate.tableA = m_alpha
    m_gate.tableB = m_alpha + m_beta
    h_alpha = per_ms * 0.07 * np.exp(-v / 20e-3)
    h_beta = per_ms * 1/(np.exp((30e-3 - v) / 10e-3) + 1)
    h_gate = moose.element('%s/gateY' % (na.path))
    h_gate.min = vmin
    h_gate.max = vmax
    h_gate.divs = vdivs
    h_gate.tableA = h_alpha
    h_gate.tableB = h_alpha + h_beta
    na.tick = -1
    return na

def make_HH_K(name = 'HH_K', parent='/library', vmin=-120e-3, vmax=40e-3, vdivs=3000):
    """Create a Hodhkin-Huxley K channel under `parent`.
    
    vmin, vmax, vdivs: voltage range and number of divisions for gate tables
    
    """
    k = moose.HHChannel('%s/%s' % (parent, name))
    k.Ek = -77e-3
    k.Xpower = 4
    v = np.linspace(vmin, vmax, vdivs+1) - EREST_ACT
    n_alpha = per_ms * (10 - v * 1e3)/(100 * (np.exp((10 - v * 1e3)/10) - 1))
    n_beta = per_ms * 0.125 * np.exp(- v * 1e3 / 80)
    n_gate = moose.element('%s/gateX' % (k.path))
    n_gate.min = vmin
    n_gate.max = vmax
    n_gate.divs = vdivs
    n_gate.tableA = n_alpha
    n_gate.tableB = n_alpha + n_beta
    k.tick = -1
    return k

def make_Chem_Oscillator( name = 'osc', parent = '/library' ):
    model = moose.Neutral( parent + '/' + name )
    compt = moose.CubeMesh( model.path + '/kinetics' )
    """
    This function sets up a simple oscillatory chemical system within
    the script. The reaction system is::

        s ---a---> a  // s goes to a, catalyzed by a.
        s ---a---> b  // s goes to b, catalyzed by a.
        a ---b---> s  // a goes to s, catalyzed by b.
        b -------> s  // b is degraded irreversibly to s.

    in sum, **a** has a positive feedback onto itself and also forms **b**.
    **b** has a negative feedback onto **a**.
    Finally, the diffusion constant for **a** is 1/10 that of **b**.
    """
    # create container for model
    diffConst = 10e-12 # m^2/sec
    motorRate = 1e-6 # m/sec
    concA = 1 # millimolar
    
    # create molecules and reactions
    a = moose.Pool( compt.path + '/a' )
    b = moose.Pool( compt.path + '/b' )
    s = moose.Pool( compt.path + '/s' )
    e1 = moose.MMenz( compt.path + '/e1' )
    e2 = moose.MMenz( compt.path + '/e2' )
    e3 = moose.MMenz( compt.path + '/e3' )
    r1 = moose.Reac( compt.path + '/r1' )

    a.concInit = 0.1
    b.concInit = 0.1
    s.concInit = 1

    moose.connect( e1, 'sub', s, 'reac' )
    moose.connect( e1, 'prd', a, 'reac' )
    moose.connect( a, 'nOut', e1, 'enzDest' )
    e1.Km = 1
    e1.kcat = 1

    moose.connect( e2, 'sub', s, 'reac' )
    moose.connect( e2, 'prd', b, 'reac' )
    moose.connect( a, 'nOut', e2, 'enzDest' )
    e2.Km = 1
    e2.kcat = 0.5

    moose.connect( e3, 'sub', a, 'reac' )
    moose.connect( e3, 'prd', s, 'reac' )
    moose.connect( b, 'nOut', e3, 'enzDest' )
    e3.Km = 0.1
    e3.kcat = 1

    moose.connect( r1, 'sub', b, 'reac' )
    moose.connect( r1, 'prd', s, 'reac' )
    r1.Kf = 0.3 # 1/sec
    r1.Kb = 0 # 1/sec

    # Assign parameters
    a.diffConst = diffConst/10
    b.diffConst = diffConst
    s.diffConst = 0
    return compt
