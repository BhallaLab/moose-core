#!/usr/bin/env python
##################################################################
#  This program is part of 'MOOSE', the
#  Messaging Object Oriented Simulation Environment.
#            Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
#  It is made available under the terms of the
#  GNU Lesser General Public License version 2.1
#  See the file COPYING.LIB for the full notice.
##################################################################
# This is the python version of rallpacks test# 1 demo for MOOSE.
# Author: Subhasis Ray
# Created: April 2007
#################################################################
import moose
import math
from util import *

SIMDT           = 50e-6                           
PLOTDT          = SIMDT * 1.0                   
SIMLENGTH       = 0.25                            
N_COMPARTMENT   = 1000                            
CABLE_LENGTH    = 1e-3                            
RA              = 1.0                             
RM              = 4.0                             
CM              = 0.01                            
EM              = -0.065                          
INJECT          = 1e-10                           
DIAMETER        = 1e-6                            
LENGTH          = CABLE_LENGTH / N_COMPARTMENT

cable = moose.Neutral('/cable')
make_compartment('/cable/c1', RA, RM, CM, EM, INJECT, DIAMETER, LENGTH)
for i in range(2,N_COMPARTMENT+1):
    current = '/cable/c'+str(i)
    previous = '/cable/c'+str(i-1)
    make_compartment( current, RA, RM, CM, EM, 0.0, DIAMETER, LENGTH )
    link_compartment(previous, current)
    print 'Connected ', previous , ' to ', current

print 'Rallpack 1 model set up'

solver = moose.HSolve('/solver')
solver.seedPath = '/cable/c1'

plot = moose.Neutral('/plot')
v1 = moose.Table('/plot/v1')
vn = moose.Table('/plot/vn')
v1.stepmode = 3
vn.stepmode = 3
context.connect(v1.id, 'inputRequest', context.pathToId('/cable/c1'), 'Vm')
context.connect(vn.id, 'inputRequest', context.pathToId('/cable/c1000'), 'Vm')

context.setClock(0, SIMDT, 0)
context.setClock(1, SIMDT, 1)
context.setClock(2, PLOTDT, 0)
# This looks ugly - find some better way, this is too much bound to GENESIS language
context.useClock(0, '/cable/##[TYPE=Compartment]')
context.useClock(1, '/solver')
context.useClock(2, '/plot/##[TYPE=Table]')
context.reset()

context.step(SIMLENGTH)
v1.dumpFile("sim_cable.0")
vn.dumpFile("sim_cable.x")
print "Plots written to 'sim_cable.*'"


