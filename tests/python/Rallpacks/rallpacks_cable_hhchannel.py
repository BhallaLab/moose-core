#!/usr/bin/env python

"""rallpacks_cable_hhchannel.py: 

Last modified: Wed May 21, 2014  09:51AM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@iitb.ac.in"
__status__           = "Development"

import sys
sys.path.append('../../../python')
import moose
from moose import utils

import os
import numpy as np
import matplotlib.pyplot as plt
import compartment as comp

EREST_ACT = -70e-3
per_ms = 1e3
dt = 5e-5

cable = []

def createChannel(species, path, **kwargs):
    """Create a channel """
    A = kwargs['A']
    B = kwargs['B']
    V0 = kwargs['V0']

def create_na_chan(parent='/library', name='na', vmin=-110e-3, vmax=50e-3, vdivs=3000):
    """Create a Hodhkin-Huxley Na channel under `parent`.
    
    vmin, vmax, vdivs: voltage range and number of divisions for gate tables
    
    """
    na = moose.HHChannel('%s/%s' % (parent, name))
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
    return na

def create_k_chan(parent='/library', name='k', vmin=-120e-3, vmax=40e-3, vdivs=3000):
    """Create a Hodhkin-Huxley K channel under `parent`.
    
    vmin, vmax, vdivs: voltage range and number of divisions for gate tables
    
    """
    k = moose.HHChannel('%s/%s' % (parent, name))
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
    return k
    
def create_hhcomp(parent='/library', name='hhcomp', diameter=1e-6, length=1e-6):
    """Create a compartment with Hodgkin-Huxley type ion channels (Na and
    K).

    Returns a 3-tuple: (compartment, nachannel, kchannel)

    """
    compPath = '{}/{}'.format(parent, name)
    mc = comp.MooseCompartment( compPath, length, diameter, {})
    c = mc.mc_
    sarea = mc.surfaceArea

    if moose.exists('/library/na'):
        moose.copy('/library/na', c.path, 'na')
    else:
        create_na_chan(parent = c.path)
    na = moose.element('%s/na' % (c.path))

    # Na-conductance 120 mS/cm^2
    na.Gbar = 120e-3 * sarea * 1e4 
    na.Ek = 115e-3 + EREST_ACT

    moose.connect(c, 'channel', na, 'channel')
    if moose.exists('/library/k'):
        moose.copy('/library/k', c.path, 'k')
    else:
        create_k_chan(parent = c.path)

    k = moose.element('%s/k' % (c.path))
    # K-conductance 36 mS/cm^2
    k.Gbar = 36e-3 * sarea * 1e4 
    k.Ek = -12e-3 + EREST_ACT
    moose.connect(c, 'channel', k, 'channel')
    return (c, na, k)

def makeCable( nsegs = 10 ):
    global cable
    moose.Neutral('/cable')
    for i in range( nsegs ):
        compName = 'hhcomp{}'.format(i)
        hhComp = create_hhcomp( '/cable', compName )
        cable.append( hhComp[0] )

    # connect the cable.
    for i, hhc in enumerate(cable[0:-1]):
        hhc.connect('axial', cable[i+1], 'raxial')


def setupDUT():
    global cable
    comp = cable[0]
    data = moose.Neutral('/data')
    pg = moose.PulseGen('/data/pg')
    pg.firstWidth = 25e-3
    pg.firstLevel = 1e-10
    moose.connect(pg, 'output', comp, 'injectMsg')
    setupClocks()
    
def setupClocks( ):
    moose.setClock(0, dt)
    moose.setClock(1, dt)

def setupSolver( hsolveDt = 5e-5 ):
    hsolvePath = '/hsolve'
    hsolve = moose.HSolve( hsolvePath )
    hsolve.dt = hsolveDt
    hsolve.target = '/cable'
    moose.useClock(1, hsolvePath, 'process')

def simulate( runTime = 25e-3 ):
    """ Simulate the cable """
    moose.useClock(0, '/cable/##', 'process')
    moose.useClock(0, '/cable/##', 'init')
    moose.useClock(1, '/##', 'process')
    moose.reinit()
    setupSolver()
    #utils.verify()
    moose.start( runTime )

if __name__ == '__main__':
    global cable
    global dt
    makeCable( 1000 )
    setupClocks()
    setupDUT()
    table0 = utils.recordAt( '/table0', cable[0], 'vm')
    table1 = utils.recordAt( '/table1', cable[-1], 'vm')
    simulate( 25e-3 )
    utils.plotTables( [ table0, table1 ], file = 'rallpack3.png', xscale = dt )
    #test_hhcomp()
