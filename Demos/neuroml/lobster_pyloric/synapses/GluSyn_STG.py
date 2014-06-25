#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import math

from pylab import *

try:
    import moose
except ImportError:
    print "ERROR: Could not import moose. Please add the directory containing moose.py in your PYTHONPATH"
    import sys
    sys.exit(1)

from moose.utils import * # for BSplineFill

class GluSyn_STG(moose.SynChan):
    """Glutamate graded synapse"""
    def __init__(self, *args):
        moose.SynChan.__init__(self,*args)
        self.Ek = -70e-3 # V
        self.Gbar = 5e-6 # S # set weight on connecting the network
        self.tau1 = 40e-3 # s # this is Vpre dependent (see below)
        self.tau2 = 0.0 # single first order equation

        Vth = -35e-3 # V
        Delta = 5e-3 # V
        ######## Graded synapse activation
        inhsyntable = moose.Interpol(self.path+"/graded_table")
        graded = moose.Mstring(self.path+'/graded')
        graded.value = 'True'
        graded = moose.Mstring(self.path+'/mgblockStr')
        graded.value = 'False'
        inhsyntable.xmin = -70e-3 # V
        inhsyntable.xmax = 0e-3 # V
        #inhsyntable.xdivs = 12
        act = [0.0] # at -70 mV
        act.extend( [1/(1+math.exp((Vth-vm)/Delta)) for vm in arange(-70e-3,0.00001e-3,70e-3/1000.)] )
        act.extend([1.0]) # at 0 mV
        #for i,activation in enumerate(act):
        #    inhsyntable[i] = activation
        #inhsyntable.tabFill(1000,BSplineFill)
        inhsyntable.vector = array(act)
        inhsyntable.connect("lookupOut",self,"activation")

        ######## Graded synapse tau
        inhtautable = moose.Interpol(self.path+"/tau_table")
        inhtautable.xmin = -70e-3 # V
        inhtautable.xmax = 0e-3 # V
        #inhtautable.xdivs = 12
        act = [0.0] # at -70 mV
        act.extend( [1/(1+math.exp((Vth-vm)/Delta)) for vm in arange(-70e-3,0.00001e-3,70e-3/1000.)] )
        act.extend([1.0]) # at 0 mV
        #for i,activation in enumerate(act):
        #    inhtautable[i] = activation
        #inhtautable.tabFill(1000,BSplineFill)
        inhtautable.vector = array(act)
        inhtautable.connect("lookupOut",self,"setTau1")
