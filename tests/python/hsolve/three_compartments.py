#!/usr/bin/env python

"""three_compartments.py: 

    A test snippet file testing for three compartments. First compartment gets
    the pulse of 0.1 nA. And we record from second and third compartment.

Last modified: Sat May 17, 2014  12:37AM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.ac.in"
__status__           = "Development"

import os
import sys
sys.path.append('../../../python')
import moose
import moose.utils as utils
import pylab

# Cable contains all compartment of moose.
cable = []

def makeCable( numberOfCompartments ):
    """ Create a cable of numberOfCompartments size"""
    global cable
    for i in range(numberOfCompartments):
        c = moose.Compartment('/cable/comp{}'.format(i))
        c.Ra = 5e7
        c.Rm = 1e11
        c.Cm = 0.015/c.Rm
        c.Em = -65e-3
        c.initVm = c.Em
        cable.append(c)

def connectCable():
    """ Connect a given cable """
    global cable
    for i, comp in enumerate( cable[:-1] ):
        comp.connect('raxial', cable[i+1], 'axial')
    setup_stim()

def setup_stim():
    pulseGen = moose.PulseGen('/data/pulse')
    pulseGen.level[0] = 1e-10
    pulseGen.width[0] = 0.25
    pulseGen.delay[0] = 0.0
    pulseGen.delay[1] = 0.50
    pulseGen.connect('output', cable[0], 'injectMsg')

def setupHSolve():
    global cable
    hsolvePath = '/hsolve'
    hsolve = moose.HSolve(hsolvePath)
    moose.useClock(1, hsolvePath, 'process')
    hsolve.dt = 1e-6
    hsolve.target = cable[0].path
    

def setupClocks():
    """Setup clock for simulation. """
    global cable
    moose.setClock(0, 1e-5)
    moose.setClock(1, 1e-6)
    moose.useClock(0, '/cable/##', 'init')
    moose.useClock(0, '/cable/##', 'process')
    moose.useClock(0, '/data/##', 'process')

def main():
    moose.Neutral('/cable')
    moose.Neutral('/data')
    makeCable(2)
    connectCable()
    table1 = utils.recordAt( '/data/table1', cable[-1], 'vm')
    setupClocks()
    moose.reinit()
    setupHSolve()
    utils.verify()
    moose.start( 0.25 )
    utils.plotTables([table1], xscale = 1e-5)
    pylab.show()

if __name__ == '__main__':
    main()
