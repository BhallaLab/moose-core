#!/usr/bin/env python

"""test_hsolve.py: 

    A script to test if hsolve is working.

Last modified: Tue Jun 03, 2014  08:33PM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
sys.path.append("../../python")
import moose

cable = []

def makeCable():
    global cable
    c1 = moose.Compartment("/compt1")
    c2 = moose.Compartment("/compt2")
    cable += [c1, c2]
    c1.connect("axial", c2, "raxial")

def setupHSolve():
    global cable
    hsolve = moose.HSolve('/hsolve')
    moose.setClock(0, 1e-5)
    moose.useClock(0, cable[0].path, 'process')

def addStimulus():
    """Stimulus """
    global cable
    c0 = cable[0]
    # PulseGen
    stim = moose.PulseGen('/pulse_gen')
    stim.connect('output', c0, 'injectMsg')

def setupTable():
    """Add tables """
    global cable
    t = moose.Table('/table')
    t.connect('requestOut', cable[0], 'getVm')

def main():
    """Main function """
    makeCable()
    setupHSolve()
    addStimulus()
    setupTable()
    moose.reinit()
    
if __name__ == '__main__':
    main()
