import moose
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

Vm = moose.Table('Vm')
Vm.stepmode = 3
Vm.connect('inputRequest', squid, 'Vm')

Na.xGate = moose.HHGate(moose.PyMooseBase.pathToId(Na.path()+'/xGate'))
Na.xGate.A = moose.InterpolationTable(moose.PyMooseBase.pathToId(Na.xGate.path()+'/A'))
Na.xGate.B = moose.InterpolationTable(moose.PyMooseBase.pathToId(Na.xGate.path()+'/B'))
Na.yGate = moose.HHGate(moose.PyMooseBase.pathToId(Na.path()+'/yGate'))
Na.yGate.A = moose.InterpolationTable(moose.PyMooseBase.pathToId(Na.yGate.path()+'/A'))
Na.yGate.B = moose.InterpolationTable(moose.PyMooseBase.pathToId(Na.yGate.path()+'/B'))
K.xGate = moose.HHGate(moose.PyMooseBase.pathToId(K.path()+'/xGate'))
K.xGate.A = moose.InterpolationTable(moose.PyMooseBase.pathToId(K.xGate.path()+'/A'))
K.xGate.B = moose.InterpolationTable(moose.PyMooseBase.pathToId(K.xGate.path()+'/B'))


Na.xGate.A.xmin = VMIN
Na.xGate.B.xmin = VMIN
Na.yGate.A.xmin = VMIN
Na.yGate.B.xmin = VMIN
K.xGate.A.xmin = VMIN
K.xGate.B.xmin = VMIN
Na.xGate.A.xmax = VMAX
Na.xGate.B.xmax = VMAX
Na.yGate.A.xmax = VMAX
Na.yGate.B.xmax = VMAX
K.xGate.A.xmax = VMAX
K.xGate.B.xmax = VMAX
Na.xGate.A.xdivs = NDIVS
Na.xGate.B.xdivs = NDIVS
Na.yGate.A.xdivs = NDIVS
Na.yGate.B.xdivs = NDIVS
K.xGate.A.xdivs = NDIVS
K.xGate.B.xdivs = NDIVS


v = VMIN


for i in range(NDIVS+1):
	Na.xGate.A[i] = calc_Na_m_A ( v )
        #print 'Na.xGate.A[',i,'] =',Na.xGate.A[i]
	Na.xGate.B [i]  =  calc_Na_m_A (v)   +  calc_Na_m_B ( v   )
        #print 'Na.xGate.B [',i,']  =',Na.xGate.B [i]
	Na.yGate.A [i] = calc_Na_h_A  (v )
        #print 'Na.yGate.A [',i,'] =',Na.yGate.A [i]
	Na.yGate.B [i] =  calc_Na_h_A  (v)   +   calc_Na_h_B  (v   )
        #print 'Na.yGate.B [',i,'] =',Na.yGate.B [i]
	K.xGate.A [i] = calc_K_n_A  (v )
        #print 'K.xGate.A [',i,'] =', K.xGate.A [i] 
	K.xGate.B [i] =  calc_K_n_A ( v)   +  calc_K_n_B ( v )
        #print 'K.xGate.B [',i,'] =', K.xGate.B [i]
	v = v + dv


# useclock /Vm,/squid,/squid/# 0
# useclock /##[TYPE=Table] 1
# // useclock /Vm 0

# // Crazy hack, but the squid demo does it and we need to match.
# setfield /squid initVm {EREST}
# reset
# setfield /squid inject 0
# step 0.005 -t
# setfield /squid inject {INJECT}
# step 0.040 -t
# setfield /squid inject 0
# step 0.005 -t
# setfield /Vm print "squid.plot"

context = moose.PyMooseBase.getContext()
context.setClock(0, SIMDT, 0)
context.setClock(1, PLOTDT, 0)
context.useClock(moose.PyMooseBase.pathToId('/sched/cj/t0'), '/Vm,/squid,/squid/#')
# Ask Upi about this comma separated wildcard path list - why does it not work when I put separate useClock statements.
# context.useClock(moose.PyMooseBase.pathToId('/sched/cj/t0'), '/Vm')
# context.useClock(moose.PyMooseBase.pathToId('/sched/cj/t0'), '/squid')
# context.useClock(moose.PyMooseBase.pathToId('/sched/cj/t0'), '/squid/#')
# context.useClock(moose.PyMooseBase.pathToId('/sched/cj/t1'), '/##[TYPE=Table]')

# setfield /squid initVm {EREST}
# reset
# setfield /squid inject 0
# step 0.005 -t
# setfield /squid inject {INJECT}
# step 0.040 -t
# setfield /squid inject 0
# step 0.005 -t
# setfield /Vm print "squid.plot"

squid.initVm = EREST
context.reset()
squid.inject = 0
context.step(0.005)
squid.inject = INJECT
context.step(0.040)
squid.inject = 0
context.step(0.005)
Vm.dumpFile("squid.plot")
# Adding some eye-candy from matplotlib - also shows the power of being
# able to combine any third-party python library with pymoose to do custom
# visualization and other funky stuff.
# Forget it (i.e. comment out) if you don't have matplotlib installed
# and be happy with plotting the 'squid.plot' output file with some
# other plotting software like xplot or gnuplot.
from pylab import plot
from pylab import array
from pylab import show
vm = array(Vm)
plot(vm)
show()
