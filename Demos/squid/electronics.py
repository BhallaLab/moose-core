# electronics.py --- 
# 
# Filename: electronics.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed Feb 22 00:53:38 2012 (+0530)
# Version: 
# Last-Updated: Tue Mar  6 16:16:11 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 185
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
        self.pulsegen = moose.PulseGen(path+"/pulse") # holding voltage/current generator
        self.pulsegen.count = 3
        self.pulsegen.firstLevel = 25.0
        self.pulsegen.firstWidth = 50.0
        self.pulsegen.firstDelay = 2.0
        self.pulsegen.secondDelay = 1e6
        self.pulsegen.trigMode = 0
        self.pulsegen.delay[2] = 1e9
        self.lowpass = moose.RC(path+"/lowpass") # lowpass filter
        self.lowpass.R = 1.0
        self.lowpass.C = 0.03
        self.vclamp = moose.DiffAmp(path+"/vclamp")
        self.vclamp.gain = 0.0
        self.vclamp.saturation = 1e10
        self.iclamp = moose.DiffAmp(path+"/iclamp")
        self.iclamp.gain = 0.0
        self.iclamp.saturation = 1e10
        self.pid = moose.PIDController(path+"/pid")
        self.pid.gain = 0.5
        self.pid.tauI = 0.02
        self.pid.tauD = 0.005
        self.pid.saturation = 1e10
        # Connect current clamp circuitry
        moose.connect(self.pulsegen, "outputOut", self.iclamp, "plusIn")
        moose.connect(self.iclamp, "outputOut", compartment, "injectMsg")
        # Connect voltage clamp circuitry
        moose.connect(self.pulsegen, "outputOut", self.lowpass, "injectIn")
        moose.connect(self.lowpass, "outputOut", self.vclamp, "plusIn")
        moose.connect(self.vclamp, "outputOut", self.pid, "commandIn")
        moose.connect(compartment, "VmOut", self.pid, "sensedIn")
        moose.connect(self.pid, "outputOut", compartment, "injectMsg")
        current_table = moose.Table("/data/Im")
        moose.connect(current_table, "requestData", compartment, "get_Im")

    def configure_pulses(self, baselevel=0.0, firstlevel=0.1, firstdelay=5.0, firstwidth=40.0, secondlevel=0.0, seconddelay=1e6, secondwidth=0.0):
        """Set up the pulse generator."""        
        self.pulsegen.baseLevel = baselevel
        self.pulsegen.firstLevel = firstlevel
        self.pulsegen.firstWidth = firstwidth
        self.pulsegen.firstDelay = firstdelay
        self.pulsegen.secondLevel = secondlevel
        self.pulsegen.secondDelay = seconddelay
        self.pulsegen.secondWidth = secondwidth
        
        
    def do_voltage_clamp(self):
        """Switch to voltage clamp circuitry. After this the simdt may
        need to be changed for correct performance."""
        self.vclamp.gain = 1.0
        self.iclamp.gain = 0.0
        self.pid.gain = 0.5
        self.pid.tauD = 0.005
        self.pid.tauI = 0.02
            
    def do_current_clamp(self):
        """Switch to current clamp circuitry. After this the simdt may
        need to be changed for correct performance."""        
        self.iclamp.gain = 1.0
        self.vclamp.gain = 0.0
        self.pid.gain = 0.0

        
# 
# electronics.py ends here
