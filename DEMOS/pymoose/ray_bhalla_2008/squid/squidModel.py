#!/usr/bin/env python

# This program creates a squid axon model along with tables to pull data
import sys
from math import *

import numpy

# The PYTHONPATH should contain the location of moose.py and _moose.so
# files.  Putting ".." with the assumption that moose.py and _moose.so
# has been generated in ${MOOSE_SOURCE_DIRECTORY}/pymoose/ (as default
# pymoose build does) and this file is located in
# ${MOOSE_SOURCE_DIRECTORY}/pymoose/examples
sys.path.append('../..')
try:
    import moose
except ImportError:
    print "ERROR: Could not import moose. Please add the directory containing moose.py in your PYTHONPATH"
    import sys
    sys.exit(1)


from squid import *

EPSILON = numpy.finfo(float).eps
class SquidModel(moose.Neutral):
    """Container for squid axon model"""
    def __init__(self, *args):
        moose.Neutral.__init__(self, *args)

        self._data = moose.Neutral("data", self)
        self._model = moose.Neutral("model", self)
        self._setupClocks()
        # Setup the compartment and table to pull data
        self._squidAxon = Squid("squidAxon", self._model)
        self._vmTable = moose.Table("vmTable", self._data)
        self._vmTable.stepMode = 3
        self._vmTable.connect("inputRequest", self._squidAxon, "Vm")
        self._gNaTable = moose.Table("gNaTable", self._data)
        self._gNaTable.stepMode = 3
        self._gNaTable.connect("inputRequest", self._squidAxon.Na(), "Gk")
        self._gKTable = moose.Table("gKTable", self._data)
        self._gKTable.stepMode = 3
        self._gKTable.connect("inputRequest", self._squidAxon.K(), "Gk")
        self._iNaTable = moose.Table("iNaTable", self._data)
        self._iNaTable.stepMode = 3
        self._iNaTable.connect("inputRequest", self._squidAxon.Na(), "Ik")
        self._iKTable = moose.Table("iKTable", self._data)
        self._iKTable.stepMode = 3
        self._iKTable.connect("inputRequest", self._squidAxon.K(), "Ik")
        self._nParamTable = moose.Table("nParamTable", self._data)
        self._nParamTable.stepMode = 3
        self._nParamTable.connect("inputRequest", self._squidAxon.K(), "X")
        self._mParamTable = moose.Table("mParamTable", self._data)
        self._mParamTable.stepMode = 3
        self._mParamTable.connect("inputRequest", self._squidAxon.Na(), "X")
        self._hParamTable = moose.Table("hParamTable", self._data)
        self._hParamTable.stepMode = 3
        self._hParamTable.connect("inputRequest", self._squidAxon.Na(), "Y")
        
        # Set up the pulsegenrator and table to pull the pulse data
        self._pulseGen = moose.PulseGen("pulseGen", self._model)
        self._setupPulseGen()
        self._pulseGen.connect("outputSrc", self._squidAxon, "injectMsg")
        self._iInjectTable = moose.Table("iInjectTable", self._data)
        self._iInjectTable.stepMode = 3
        self._iInjectTable.connect("inputRequest", self._pulseGen, "output")
        # Assign clock ticks
        self.getContext().useClock(0, self._model.path+"/##")
        self._squidAxon.useClock(1, "init")
        self.getContext().useClock(2,self._data.path+"/#")
        self._runTime = 0.050

    def _setupPulseGen(self, paramDict=None):
        self._pulseGen.trigMode = 0

        if paramDict == None: # Default values
            self._pulseGen.firstLevel = 0.1e-6
            self._pulseGen.firstDelay = 0.005
            self._pulseGen.firstWidth = 0.040
            self._pulseGen.secondLevel = 0.0
            self._pulseGen.secondDelay = 1e8
            self._pulseGen.secondWidth = 0.0
        else:
            self._pulseGen.firstLevel = paramDict["firstLevel"]
            self._pulseGen.firstWidth = paramDict["firstWidth"]
            self._pulseGen.firstDelay = paramDict["firstDelay"]
            self._pulseGen.secondLevel = paramDict["secondLevel"]
            self._pulseGen.secondWidth = paramDict["secondWidth"]
            self._pulseGen.secondDelay = paramDict["secondDelay"]
            self._pulseGen.baseLevel = paramDict["baseLevel"]
            if paramDict["singlePulse"]:
                self._pulseGen.trigMode = 1
            #                 self._pulseGen.secondDelay = 1e8 # Set a large delay to prevent a second pulse within the simulation time
            else:
                self._pulseGen.trigMode = 0

            print "firstLevel:", self._pulseGen.firstLevel, \
                "firstWidth:", self._pulseGen.firstWidth, \
                "firstDelay:", self._pulseGen.firstDelay, \
                "secondLevel:", self._pulseGen.secondLevel, \
                "secondWidth:",self._pulseGen.secondWidth, \
                "secondDelay:", self._pulseGen.secondDelay, \
                "baseLevel:", self._pulseGen.baseLevel

    def _setupClocks(self, simDt=None, plotDt=None):
        if simDt == None:
            simDt = 1e-6
        if plotDt == None:
            plotDt = 1e-4
        self.getContext().setClock(0, simDt, 0)
        if not hasattr(self, "_clockTick0"):
            self._clockTick0 = moose.ClockTick("/sched/cj/t0")
        self.getContext().setClock(1, simDt, 1)
        if not hasattr(self, "_clockTick1"):
            self._clockTick1 = moose.ClockTick("/sched/cj/t1")
        self.getContext().setClock(2, plotDt, 0)
        if not hasattr(self, "_clockTick2"):
            self._clockTick2 = moose.ClockTick("/sched/cj/t2")

    def vmTable(self):
        return self._vmTable

    def gNaTable(self):
        return self._gNaTable

    def gKTable(self):
        return self._gKTable

    def iNaTable(self):
        return self._iNaTable

    def iKTable(self):
        return self._iKTable

    def nParamTable(self):
        return self._nParamTable

    def mParamTable(self):
        return self._mParamTable

    def hParamTable(self):
        return self._hParamTable

    def iInjectTable(self):
        return self._iInjectTable

    def pulseGen(self):
        return self._pulseGen

    def simDt(self):
        return self._clockTick0.dt

    def plotDt(self):
        return self._clockTick2.dt

    def setSimDt(self, simDt):
        self.getContext().setClock(0, simDt, 0)
        self.getContext().setClock(1, simDt, 1)

    def setPlotDt(self, plotDt):
        self.getContext().setClock(2, plotDt, 0)

    def runTime(self):
        return self._runTime


    def doResetForIClamp(self, paramDict=None):
        """Do setup for current clamp and do a reset.
        paramDict should contain the following keys:
        simDt - simulation time step (+ve float)
        plotDt - plotting time step (+ve float < simDt)
        firstLevel - amplitude of injection current in first pulse
        firstWidth - time for which first current injection is applied
        firstDelay - time from start of simulation when the first pulse should be applied
        secondLevel - amplitude of second current injection
        secondDelay - time from the beginning of fisrt pulse when second pulse is to be applied
        secondWidth - duration of second injection current
        runTime - total simulation time - must be greater than the total duration of the current injection
        """
        if not paramDict == None:
            simDt = paramDict["simDt"]
            plotDt = paramDict["plotDt"]
            if plotDt < 0.0:
                print "Plot time step must be positive"
            if simDt < 0.0:
                print "Simulatipon time step must be postive"
            if fabs(self._clockTick2.dt - plotDt) > EPSILON:
                self.setPlotDt(plotDt)
            if fabs(self._clockTick0.dt - simDt) > EPSILON:
                self.setSimDt(simDt)
            self._setupPulseGen(paramDict)
            self._squidAxon.blockNaChannel(paramDict["blockNa"])
            self._squidAxon.blockKChannel(paramDict["blockK"])
            self._squidAxon.setIonPotential(paramDict["temperature"], paramDict["naConc"], paramDict["kConc"])
#             self._runTime = paramDict["runTime"]
        self.getContext().reset()
        if paramDict["singlePulse"]: # This is an ugly hack borrowed from GENESIS squid demo
            self._pulseGen.trigTime = 0.0 
        # !SquidModel.doResetForIClamp

    def doRun(self, runTime=None):
        if not runTime == None:
            self._runTime = runTime
        self.getContext().step(self._runTime)
        
    def dumpPlotData(self):
        self._vmTable.dumpFile("squidModelVm.plot")
        print "SquidModel.dumpPlotData() - Vm is in squidModelVm.plot"
        self._iInjectTable.dumpFile("squidModelIInject.plot")
        print "SquidModel.dumpPlotData() - Injection current is in squidModelIInject.plot"
        self._iNaTable.dumpFile("squidModelINa.plot")
        print "SquidModel.dumpPlotData() - INa is in squidModelINa.plot"
        self._iKTable.dumpFile("squidModelIK.plot")
        print "SquidModel.dumpPlotData() - IK is in squidModelIK.plot"
        self._gNaTable.dumpFile("squidModelGNa.plot")
        print "SquidModel.dumpPlotData() - GNa is in squidModelGNa.plot"
        self._gKTable.dumpFile("squidModelGK.plot")
        print "SquidModel.dumpPlotData() - GK is in squidModelGK.plot"
        

def testSquidModel():
    squidModel = SquidModel("/testSquidModel")
    squidModel.doResetForIClamp()
    squidModel.doRun()
    squidModel.dumpPlotData()

if __name__ == "__main__":
    testSquidModel()
    
    
