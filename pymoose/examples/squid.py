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

context = moose.PyMooseBase.getContext()

class Squid(moose.Compartment):
    """A squid giant axon compartment - initialized to default property values"""
    def __init__(self, path):
        """Set up the squid giant axon model with defaults"""
        global context
        moose.Compartment.__init__(self,path)

        self.vmTable = moose.Table('Vm')
        self.Rm = RM
        self.Ra = RA
        self.Cm = CM
        self.Em = VLEAK
        self.initVm = EREST
        self.inject = 0
        self.Na =  moose.HHChannel('Na', self)
        self.Na.Ek = VNa
        self.Na.Gbar = GNa
        self.Na.Xpower = 3 # This will create HHGate instance xGate inside the Na channel
        self.Na.Ypower = 1 # This will create HHGate instance yGate inside the Na channel
        self.K = moose.HHChannel('K', self)
        self.K.Ek = VK
        self.K.Gbar = GK
        self.K.Xpower = 4 # This will create HHGate instance xGate inside the K channel
        self.connect('channel', self.Na, 'channel')
        self.connect('channel', self.K, 'channel')
        # These gates get created only after Xpower or Ypower are set to nonzero values
        # So we have to explicitly insert these fields in the class
        self.Na.xGate = moose.HHGate(self.Na.path() + '/xGate')
        self.Na.yGate = moose.HHGate(self.Na.path() + '/yGate')
        self.K.xGate = moose.HHGate(self.K.path() + '/xGate')
        self.K.yGate = moose.HHGate(self.K.path() + '/yGate')

        self.Na.xGate.A.xmin = VMIN
        self.Na.xGate.A.xmax = VMAX
        self.Na.xGate.A.xdivs = NDIVS


        self.Na.xGate.B.xmin = VMIN
        self.Na.xGate.B.xmax = VMAX
        self.Na.xGate.B.xdivs = NDIVS


        self.Na.yGate.A.xmin = VMIN
        self.Na.yGate.A.xmax = VMAX
        self.Na.yGate.A.xdivs = NDIVS

        self.Na.yGate.B.xmin = VMIN
        self.Na.yGate.B.xmax = VMAX
        self.Na.yGate.B.xdivs = NDIVS

        self.K.xGate.A.xmin = VMIN
        self.K.xGate.A.xmax = VMAX
        self.K.xGate.A.xdivs = NDIVS
        
        self.K.xGate.B.xmin = VMIN
        self.K.xGate.B.xmax = VMAX
        self.K.xGate.B.xdivs = NDIVS

        v = VMIN

        for i in range(NDIVS+1):
            self.Na.xGate.A[i] = calc_Na_m_A(v)
            self.Na.xGate.B[i] = calc_Na_m_A(v) + calc_Na_m_B(v)
            self.Na.yGate.A[i] = calc_Na_h_A(v)
            self.Na.yGate.B[i] = calc_Na_h_A(v) + calc_Na_h_B(v)
            self.K.xGate.A[i]  = calc_K_n_A(v)
            self.K.xGate.B[i]  = calc_K_n_A(v) + calc_K_n_B(v)
            v = v + dv

        context = moose.PyMooseBase.getContext()
        context.setClock(0, SIMDT, 0)
        context.setClock(1, PLOTDT, 0)
        print 'Inside the constructor'
        # Create and connect a table to dump data
        self.vmTable.stepmode = 3
        self.vmTable.connect('inputRequest', self, 'Vm')
        clockTargets ='/Vm,'+self.path()+','+self.path()+'/#'
        print 'Clock targets = ', clockTargets
        context.useClock(moose.PyMooseBase.pathToId('/sched/cj/t0'), clockTargets)


        
    def doRun(self, totTime,preTime, pulseWidth, injectBase, injectHigh):
        global context
        if ( preTime > totTime ):
            print "ERROR: total simulation time is less than pre-pulse time."
            return self.vmTable
        elif preTime + pulseWidth > totTime:
            pulseWidth = totTime - preTime
            cout << "Warning: Pulse width exceeds total simulation time."
        self.initVm = EREST
        context.reset()
        self.inject = injectBase
        time = preTime
        context.step(time)
        self.inject = injectHigh
        time = pulseWidth
        context.step(time)
        self.inject = injectBase
        time = totTime - preTime - pulseWidth
        if  time > 0 :
            context.step(time)
        return self.vmTable

if __name__ == '__main__':
    squid = Squid('/squid')
    vm = squid.doRun(0.050, 0.005, 0.040, 0, INJECT)
    for i in vm:
        print 'vm = ',i
    vm.dumpFile('squid.plot')
    print 'The output is available in squid.plot'
