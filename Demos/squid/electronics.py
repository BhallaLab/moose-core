# electronics.py --- 
# 
# Filename: electronics.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed Feb 22 00:53:38 2012 (+0530)
# Version: 
# Last-Updated: Wed Feb 22 23:38:14 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 136
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 2012-02-22 23:22:30 (+0530) Subha - the circuitry put in a class.
# 

# Code:

import numpy
import moose

class ClampCircuit(moose.Neutral):
    """Container for a Voltage-Clamp/Current clamp circuit."""
    def __init__(self, path, compartment):
        moose.Neutral.__init__(self, path)
        self.pulsegen = moose.PulseGen(path+'/pulse') # holding voltage/current generator
        self.lowpass = moose.RC(path+'/lowpass') # lowpass filter
        self.vclamp = moose.DiffAmp(path+'/vclamp')
        self.iclamp = moose.DiffAmp(path+'/iclamp')
        self.pid = moose.PIDController(path+'/pid')
        self.pulsegen.firstLevel = 25.0
        self.pulsegen.firstWidth = 50.0
        self.pulsegen.firstDelay = 2.0
        self.pulsegen.secondDelay = 1e6
        self.pulsegen.trigMode = 0
        self.lowpass.R = 1.0
        self.lowpass.C = 0.03
        self.iclamp.gain = 0.0
        self.vclamp.gain = 0.0
        self.pid.gain = 0.5
        self.pid.tauI = 0.02
        self.pid.tauD = 0.005
        self.pid.saturation = 1e10
        # Connect current clamp circuitry
        self.pulsegen.connect("outputOut", self.iclamp, "plusIn")
        self.iclamp.connect("outputOut", compartment, "injectMsg")
        # Connect voltage clamp circuitry
        self.pulsegen.connect("outputOut", self.lowpass, "injectIn")
        self.lowpass.connect("outputOut", self.vclamp, "plusIn")
        self.vclamp.connect("outputOut", self.pid, "commandIn")
        compartment.connect("VmOut", self.pid, "sensedIn")
        moose.connect(self.pid, "outputOut", compartment, "injectMsg")
        # TODO: setup tables for recording current and voltage clamp current.
        # injections We have to maintain two tables as MOOSE does not
        # report the correct Compartment injection current.

    def configure_pulses(self, firstlevel=0.1, firstdelay=5.0, firstwidth=40.0, secondlevel=0.0, seconddelay=1e6, secondwidth=0.0):
        self.pulsegen.firstLevel = firstlevel
        self.pulsegen.firstWidth = firstwidth
        self.pulsegen.firstDelay = firstdelay
        self.pulsegen.secondLevel = secondlevel
        self.pulsegen.secondDelay = seconddelay
        self.pulsegen.secondWidth = secondwidth        
        
    def do_voltage_clamp(self, simdt):
        self.lowpass.R = 1.0 # KOhm
        self.lowpass.C = 0.003 # uF
        self.vclamp.gain = 1.0
        self.iclamp.gain = 0.0
        self.pid.gain = 0.7e-6
        self.pid.tauD = simdt
        self.pid.tauI = 2 * simdt
            
    def do_current_clamp(self):
        self.iclamp.gain = 1.0
        self.vclamp.gain = 0.0
        self.pid.gain = 0.0

        
# 
# electronics.py ends here
