#!/usr/bin/env python
import sys
sys.path.append('..')
try:
    import moose
except ImportError:
    print "ERROR: Could not import moose. Please add the directory containing moose.py in your PYTHONPATH"
    import sys
    sys.exit(1)

RUNTIME = 200.0
SIMDT = 1.0

# Pulse generator with trigger mode = 0 This is free running - and
# creates a series of pulses
pulse0 = moose.PulseGen("/pulse0")
pulse0.firstLevel = 50.0
pulse0.firstWidth = 3.0
pulse0.firstDelay = 5.0
pulse0.secondLevel = -20.0
pulse0.secondWidth = 5.0
pulse0.secondDelay = 8.0
pulse0.baseLevel = 10.0
pulse0.trigMode = 0

trig = moose.PulseGen("/trig")
trig.firstLevel = 20.0
trig.firstWidth = 1.0
trig.firstDelay = 5.0
trig.secondWidth = 30.0

# Pulse generator with trigger mode = 1
pulse1 = moose.PulseGen("/pulse1")
pulse1.firstLevel = 50.0
pulse1.firstWidth = 3.0
pulse1.firstDelay = 5.0
pulse1.secondLevel = -20.0
pulse1.secondWidth = 5.0
pulse1.secondDelay = 8.0
pulse1.baseLevel = 10.0
pulse1.trigMode = 1
trig.connect("outputSrc", pulse1, "input")

# Gated pulse
gate = moose.PulseGen("/gate")
gate.firstLevel = 20.0
gate.firstWidth = 30.0
gate.firstDelay = 15.0
gate.secondWidth = 30.0

# Pulse generator with trigger mode = 2
pulse2 = moose.PulseGen("/pulse2")
pulse2.firstLevel = 50.0
pulse2.firstWidth = 3.0
pulse2.firstDelay = 5.0
pulse2.secondLevel = -20.0
pulse2.secondWidth = 5.0
pulse2.secondDelay = 8.0
pulse2.baseLevel = 10.0
pulse2.trigMode = 2
gate.connect("outputSrc", pulse2, "input")

plot0 = moose.Table("/plot0")
plot0.stepMode = 3
plot0.connect("inputRequest", pulse0, "output")

plot1 = moose.Table("/plot1")
plot1.stepMode = 3
plot1.connect("inputRequest", pulse1, "output")

plot2 = moose.Table("/plot2")
plot2.stepMode = 3
plot2.connect("inputRequest", pulse2, "output")

plotGate = moose.Table("/plotGate")
plotGate.stepMode = 3
plotGate.connect("inputRequest", gate, "output")

plotTrig = moose.Table("/plotTrig")
plotTrig.stepMode = 3
plotTrig.connect("inputRequest", trig, "output")
context = moose.PyMooseBase.getContext()

context.useClock(0, "/#[TYPE=PulseGen]")
context.useClock(1, "/#[TYPE=Table]")
context.setClock(0, SIMDT)
context.setClock(1, SIMDT)
context.reset()
context.step(RUNTIME)

plot0.dumpFile("pulse0.plot")
plot1.dumpFile("pulse1.plot")
plot2.dumpFile("pulse2.plot")
plotGate.dumpFile("gate.plot")
plotTrig.dumpFile("trig.plot")
print "pulsegen.py: finished simulation"

