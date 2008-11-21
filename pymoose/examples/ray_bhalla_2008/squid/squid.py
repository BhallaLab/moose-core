#!/usr/bin/env python
import sys
import math

# The PYTHONPATH should contain the location of moose.py and _moose.so
# files.  Putting ".." with the assumption that moose.py and _moose.so
# has been generated in ${MOOSE_SOURCE_DIRECTORY}/pymoose/ (as default
# pymoose build does) and this file is located in
# ${MOOSE_SOURCE_DIRECTORY}/pymoose/examples
sys.path.append('..\..')
try:
    import moose
except ImportError:
    print "ERROR: Could not import moose. Please add the directory containing moose.py in your PYTHONPATH"
    import sys
    sys.exit(1)


GAS_CONST = 8.314
FARADAY = 9.65e4

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
# GK = 36e-3
# GNa = 120e-3
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

class Squid(moose.Compartment):
    """A squid giant axon compartment - initialized to default property values"""
    def __init__(self, *args):
        """Set up the squid giant axon model with defaults"""
        moose.Compartment.__init__(self,*args)
        self.Rm = RM
        self.Ra = RA
        self.Cm = CM
        self.Em = VLEAK
        self.initVm = EREST
        self.inject = 0
        self._innerNaConc = 70.96
        self._innerKConc = 301.4
        self._Na =  moose.HHChannel("Na", self)
        self._Na.Ek = VNa
        self._Na.Gbar = GNa
        self._Na.Xpower = 3 # This will create HHGate instance xGate inside the Na channel
        self._Na.Ypower = 1 # This will create HHGate instance yGate inside the Na channel
        self._K = moose.HHChannel("K", self)
        self._K.Ek = VK
        self._K.Gbar = GK
        self._K.Xpower = 4 # This will create HHGate instance xGate inside the K channel
        self.connect("channel", self._Na, "channel")
        self.connect("channel", self._K, "channel")
        # These gates get created only after Xpower or Ypower are set to nonzero values
        # So we have to explicitly insert these fields in the class
        self._Na.xGate = moose.HHGate(self._Na.path + "/xGate")
        self._Na.yGate = moose.HHGate(self._Na.path + "/yGate")
        self._K.xGate = moose.HHGate(self._K.path + "/xGate")
        self._K.yGate = moose.HHGate(self._K.path + "/yGate")
        self._Na.xGate.A.xmin = VMIN
        self._Na.xGate.A.xmax = VMAX
        self._Na.xGate.A.xdivs = NDIVS
        self._Na.xGate.B.xmin = VMIN
        self._Na.xGate.B.xmax = VMAX
        self._Na.xGate.B.xdivs = NDIVS
        self._Na.yGate.A.xmin = VMIN
        self._Na.yGate.A.xmax = VMAX
        self._Na.yGate.A.xdivs = NDIVS
        self._Na.yGate.B.xmin = VMIN
        self._Na.yGate.B.xmax = VMAX
        self._Na.yGate.B.xdivs = NDIVS
        self._K.xGate.A.xmin = VMIN
        self._K.xGate.A.xmax = VMAX
        self._K.xGate.A.xdivs = NDIVS
        self._K.xGate.B.xmin = VMIN
        self._K.xGate.B.xmax = VMAX
        self._K.xGate.B.xdivs = NDIVS

        v = VMIN

        for i in range(NDIVS+1):
            self._Na.xGate.A[i] = calc_Na_m_A(v)
            self._Na.xGate.B[i] = calc_Na_m_A(v) + calc_Na_m_B(v)
            self._Na.yGate.A[i] = calc_Na_h_A(v)
            self._Na.yGate.B[i] = calc_Na_h_A(v) + calc_Na_h_B(v)
            self._K.xGate.A[i]  = calc_K_n_A(v)
            self._K.xGate.B[i]  = calc_K_n_A(v) + calc_K_n_B(v)
            v = v + dv
        
    def Na(self):
        return self._Na

    def K(self):
        return self._K

    def blockNaChannel(self, arg):
        """Block the Na channel if args is True, unblock otherwise"""
        if arg:
            self._Na.Gbar = 0.0
        else:
            self._Na.Gbar = GNa

    def blockKChannel(self, arg):
        """Block the K channel if args is True, unblock otherwise"""
        if arg:
            self._K.Gbar = 0.0
        else:
            self._K.Gbar = GK

    def setIonPotential(self, temperature, naConc, kConc):
        print "Setting [K] =", kConc, ", [Na] =", naConc
        try:
            self._Na.Ek = (GAS_CONST*temperature/FARADAY) * math.log(naConc/self._innerNaConc) + 0.07 + EREST
            self._K.Ek =  (GAS_CONST*temperature/FARADAY) * math.log(kConc/self._innerKConc) + 0.07 + EREST
        except OverflowError:
            raise OverflowError("Cannot handle 0 conc of ion.")
 
def testSquid():
    model = moose.Neutral("/model")
    data = moose.Neutral("/data")
    squid = Squid("/model/testSquid")
    vmTable = moose.Table("/data/testVm")
    gNaTable = moose.Table("/data/testGNa")
    gKTable = moose.Table("/data/testGK")
    iNaTable = moose.Table("/data/testINa")
    iKTable = moose.Table("/data/testIK")
    nTable = moose.Table("/data/testN")
    vmTable.stepMode = 3
    gNaTable.stepMode = 3
    gKTable.stepMode = 3
    iNaTable.stepMode = 3 
    iKTable.stepMode = 3
    nTable.stepMode = 3
    vmTable.connect("inputRequest", squid, "Vm")
    gNaTable.connect("inputRequest", squid.Na(), "Gk")
    gKTable.connect("inputRequest", squid.K(), "Gk")
    iNaTable.connect("inputRequest", squid.Na(), "Ik")
    iKTable.connect("inputRequest", squid.K(), "Ik")
    nTable.connect("inputRequest", squid.K(), "X")
    squid.getContext().setClock(0, SIMDT, 0)
    squid.getContext().setClock(1, SIMDT, 1)
    squid.getContext().setClock(2, PLOTDT, 0)
    squid.useClock(0)
    squid.useClock(1, "init")
    vmTable.useClock(2)
    gNaTable.useClock(2)
    gKTable.useClock(2)
    iNaTable.useClock(2) 
    iKTable.useClock(2)
    nTable.useClock(2)
    squid.getContext().reset()
    squid.inject = 0.0
    squid.getContext().step(0.005)
    squid.inject = 0.1e-6
    squid.getContext().step(0.040)
    squid.inject = 0.0
    squid.getContext().step(0.005)

    vmTable.dumpFile("testSquidVm.plot")
    gNaTable.dumpFile("testSquidGNa.plot")
    gKTable.dumpFile("testSquidGK.plot")
    iNaTable.dumpFile("testSquidINa.plot")
    iKTable.dumpFile("testSquidIK.plot")
    nTable.dumpFile("testSquidN.plot")

if __name__ == "__main__":
    testSquid()
    print "Finished test simulation"
