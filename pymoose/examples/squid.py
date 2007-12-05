#!/usr/bin/env python
import sys
import math
# The PYTHONPATH should contain the location of moose.py and _moose.so
# files.  Putting ".." with the assumption that moose.py and _moose.so
# has been generated in ${MOOSE_SOURCE_DIRECTORY}/pymoose/ (as default
# pymoose build does) and this file is located in
# ${MOOSE_SOURCE_DIRECTORY}/pymoose/examples
sys.path.append('..')
try:
    import moose
except ImportError:
    print "ERROR: Could not import moose. Please add the directory containing moose.py in your PYTHONPATH"
    import sys
    sys.exit(1)

import math
VMIN = -0.1
VMAX = 0.05
NDIVS = 150
v = VMIN
dv = ( VMAX - VMIN ) / NDIVS
SIMDT = 1e-5
PLOTDT = 1e-4
RUNTIME = 0.5
EREST = -0.07
VLEAK = EREST + 0.010613
VK = EREST -0.012
VNa = EREST + 0.115
RM = 424.4e3
RA = 7639.44e3
GLEAK = 0.3e-3
GK = 36e-3
GNa = 120e-3
CM = 0.007854e-6
INJECT = 0.1e-6

GK = 0.282743e-3
GNa = 0.94248e-3

def calc_Na_m_A( v ):
    if math.fabs(EREST+0.025-v) < 1e-6:
        v = v + 1e-6
    return 0.1e6 * (EREST + 0.025 -v)/(math.exp((EREST + 0.025 - v)/0.01) - 1.0)

def calc_Na_m_B(v):
    return 4.0e3 * math.exp((EREST - v)/0.018)

def calc_Na_h_A( v ):
    return  70.0 * math.exp(( EREST - v )/0.020 )

def calc_Na_h_B( v ):
    return ( 1.0e3 / (math.exp ( ( 0.030 + EREST - v )/ 0.01 ) + 1.0 ))

def calc_K_n_A( v ):
    if math.fabs( 0.01 + EREST - v )  < 1.0e-6 :
        v = v + 1.0e-6
    return ( 1.0e4 * ( 0.01 + EREST - v ) )/(math.exp(( 0.01 + EREST - v )/0.01) - 1.0 )

def calc_K_n_B( v ):
	return 0.125e3 * math.exp((EREST - v ) / 0.08 )

# Make the squid object global

context = moose.PyMooseBase.getContext()

def setupModel():
    """Set up the MOOSE model for squid demo"""
    global context

    if context.exists('squid'):
        return moose.Compartment('squid')
    squid = moose.Compartment('squid')
    squid.Rm = RM
    squid.Ra = RA
    squid.Cm = CM
    squid.Em = VLEAK

    Na =  moose.HHChannel('Na', squid)
    Na.Ek = VNa
    Na.Gbar = GNa
    Na.Xpower = 3
    Na.Ypower = 1
    
    K = moose.HHChannel('K', squid)
    K.Ek = VK
    K.Gbar = GK
    K.Xpower = 4

    squid.connect('channel', Na, 'channel')
    squid.connect('channel', K, 'channel')
    
    
    Na_xA = moose.InterpolationTable('/squid/Na/xGate/A')
    Na_xB = moose.InterpolationTable('/squid/Na/xGate/B')
    Na_yA = moose.InterpolationTable('/squid/Na/yGate/A')
    Na_yB = moose.InterpolationTable('/squid/Na/yGate/B')
    K_xA = moose.InterpolationTable('/squid/K/xGate/A')
    K_xB = moose.InterpolationTable('/squid/K/xGate/B')

   
    Na_xA.xmin = VMIN
    Na_xA.xmax = VMAX
    Na_xA.xdivs = NDIVS

    Na_xB.xmin = VMIN
    Na_xB.xmax = VMAX
    Na_xB.xdivs = NDIVS

    Na_yA.xmin = VMIN
    Na_yA.xmax = VMAX
    Na_yA.xdivs = NDIVS

    Na_yB.xmin = VMIN
    Na_yB.xmax = VMAX
    Na_yB.xdivs = NDIVS

    K_xA.xmin = VMIN
    K_xA.xmax = VMAX
    K_xA.xdivs = NDIVS

    K_xB.xmin = VMIN
    K_xB.xmax = VMAX
    K_xB.xdivs = NDIVS
    
    v = VMIN

    for i in range(NDIVS+1):
	Na_xA[i] = calc_Na_m_A ( v )
	Na_xB[i]  =  calc_Na_m_A (v)   +  calc_Na_m_B ( v   )
	Na_yA[i] = calc_Na_h_A  (v )
	Na_yB[i] =  calc_Na_h_A  (v)   +   calc_Na_h_B  (v   )
	K_xA[i] = calc_K_n_A  (v )
	K_xB[i] =  calc_K_n_A ( v)   +  calc_K_n_B ( v )
	v = v + dv

    context = moose.PyMooseBase.getContext()
    context.setClock(0, SIMDT, 0)
    context.setClock(1, PLOTDT, 0)


    squid.initVm = EREST
    return squid


def runDemo():
    """Run the simulation steps, setup the model if required"""
    Vm = moose.Table('Vm')
    Vm.stepmode = 3
    squid = setupModel()
    Vm.connect('inputRequest', squid, 'Vm')
    context.useClock(moose.PyMooseBase.pathToId('/sched/cj/t0'), '/Vm,/squid,/squid/#')
    context.reset()
    squid.inject = 0
    context.step(0.005)
    squid.inject = INJECT
    context.step(0.040)
    squid.inject = 0
    context.step(0.005)
    Vm.dumpFile("squid.plot")
    return Vm

if __name__ == '__main__':
    runDemo()
    print 'The output is available in squid.plot'
