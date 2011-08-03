import moose
import sys
from moosehandler import MooseHandler

class mooseElectrodes():

    def currentClamp(self,paramDict,compartment):
        self.updateTimeStepInfo()
        self._pulseGen = moose.PulseGen("PulseGen",compartment)
        self._setupPulseGen(paramDict)
        self._iClamp =  moose.DiffAmp("IClamp",compartment)
        self._iClamp.gain = 1.0
        
        # Connect current clamp circuitry
        self._pulseGen.connect("outputSrc", self._iClamp, "plusDest")
        self._iClamp.connect("outputSrc", compartment, "injectMsg")
        

    def voltageClamp(self,paraDict,compartment):
        self.updateTimeStepInfo()
        self._pulseGen = moose.PulseGen("PulseGen",compartment)
        self._setupPulseGen(paraDict)
        self._setupElectronics(compartment) #voltage clamp circuitary

    def updateTimeStepInfo(self):
        self.simDt = MooseHandler.simdt
        self.plotDt = MooseHandler.plotdt
        self.runTime = MooseHandler.runtime

   
    def _setupElectronics(self,compartment):
        self._model = compartment
        self._lowpass = moose.RC("lowpass", self._model)
        self._vClamp = moose.DiffAmp("Vclamp", self._model)
        self._PID = moose.PIDController("PID", self._model)
        #self._pulseGen.firstLevel = 25e-6
        #self._pulseGen.firstWidth = 50e-3
        #self._pulseGen.firstDelay = 2e-3
        #self._pulseGen.secondDelay = 1e6
        self._pulseGen.trigMode = 1   #?
        self._pulseGen.trigTime = 0.0 #?

        self._lowpass.R = 1.0
        self._lowpass.C = 3e-2
        self._vClamp.gain = 0.0
        self._PID.gain = 0.5e-6
        self._PID.tauI = self.simDt
        self._PID.tauD = self.simDt/4.0
        self._PID.saturation = 1e10
        
        # Connect voltage clamp circuitry
        self._pulseGen.connect("outputSrc", self._lowpass, "injectMsg")
        self._lowpass.connect("outputSrc", self._vClamp, "plusDest")
        #self._rcTable = moose.Table('RCTable', self._data)
        #self._rcTable.stepMode = 3
        #self._rcTable.connect('inputRequest', self._lowpass, 'state')
        self._vClamp.connect("outputSrc", self._PID, "commandDest")
        self._model.connect("VmSrc", self._PID, "sensedDest")
        self._PID.connect("outputSrc", self._model, "injectMsg")
        
        # Tables for recording current and voltage clamp current.
        # injections We have to maintain two tables as MOOSE does not
        # report the correct Compartment injection current.
        # self._iClampInjectTable = moose.Table('IClampInject', self._data)
        # self._iClampInjectTable.stepMode = 3
        # self._iClampInjectTable.connect("inputRequest", self._iClamp, "output")
        # self._vClampInjectTable = moose.Table('VClampInject', self._data)
        # self._vClampInjectTable.stepMode = 3
        # self._vClampInjectTable.connect("inputRequest", self._PID, "output")
        
    def _setupPulseGen(self, paramDict=None):
        self._pulseGen.trigMode = 0

        if paramDict is None: # Default values
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
            #if paramDict["singlePulse"]:
            #    self._pulseGen.trigMode = 1
                
            print "firstLevel:", self._pulseGen.firstLevel, \
                "firstWidth:", self._pulseGen.firstWidth, \
                "firstDelay:", self._pulseGen.firstDelay, \
                "secondLevel:", self._pulseGen.secondLevel, \
                "secondWidth:",self._pulseGen.secondWidth, \
                "secondDelay:", self._pulseGen.secondDelay, \
                "baseLevel:", self._pulseGen.baseLevel

    def doResetForVClamp(self, paramDict=None):
        """Do setup for voltage clamp and do a reset. 
        """
        # if paramDict is not None:
        #     simDt = paramDict["simDt"]
        #     plotDt = paramDict["plotDt"]
        #     if plotDt < 0.0:
        #         print "Plot time step must be positive"
        #     if simDt < 0.0:
        #         print "Simulatipon time step must be postive"
        #     if fabs(self._clockTick2.dt - plotDt) > EPSILON:
        #         self.setPlotDt(plotDt)
        #     if fabs(self._clockTick0.dt - simDt) > EPSILON:
        #         self.setSimDt(simDt)
        #     self._setupPulseGen(paramDict)
        #     self._squidAxon.blockNaChannel(paramDict["blockNa"])
        #     self._squidAxon.blockKChannel(paramDict["blockK"])
        #     self._squidAxon.setIonPotential(paramDict["temperature"], paramDict["naConc"], paramDict["kConc"])
        self._lowpass.R = 1.0
        self._lowpass.C = 0.03e-4
        self._vClamp.gain = 1.0
        self._PID.gain = 0.7e-6
        self._PID.tauI = self.simDt * 2
        self._PID.tauD = self.simDt
        print 'PID tauD', self._PID.tauD
        #self.getContext().reset()
        if paramDict is not None and paramDict["singlePulse"]:
            self._pulseGen.trigMode = 1
            self._pulseGen.trigTime = 0.0

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
        if paramDict is not None:
            simDt = paramDict["simDt"]
            plotDt = paramDict["plotDt"]
            # if plotDt < 0.0:
            #     print "Plot time step must be positive"
            # if simDt < 0.0:
            #     print "Simulatipon time step must be postive"
            # if fabs(self._clockTick2.dt - plotDt) > EPSILON:
            #     self.setPlotDt(plotDt)
            # if fabs(self._clockTick0.dt - simDt) > EPSILON:
            #     self.setSimDt(simDt)
            # self._setupPulseGen(paramDict)
            
        else:
            #self._setupClocks()
            self._setupPulseGen(paramDict)

        self._iClamp.gain = 1.0
        
        if paramDict is not None and paramDict["singlePulse"]:
            self._pulseGen.trigMode = 1
            self._pulseGen.trigTime = 0.0
