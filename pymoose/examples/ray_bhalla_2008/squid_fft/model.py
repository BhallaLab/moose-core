import moose

from squid import *

mean_injection_current = 0.1e-6 # in Ampere
noise_variance = 0.1 # percentage of the mean
class NoisySquid(moose.Neutral):
    """A model for recording from squid giant axon compartment with
current injection that is white noise"""
    def __init__(self, *args):
        global mean_injection_current
        moose.Neutral.__init__(self, *args)
        self.model = moose.Neutral("model", self)
        self.squidCompartment = Squid("squid", self.model)
        self.currentInjection = moose.UniformRng("injection", self.model)
	self.currentInjection.min = 0.0
	self.currentInjection.max = 0.091e-6
        self.getContext().reset()

        self.currentInjection.connect("output", self.squidCompartment,\
                                          "inject")
        self.data = moose.Neutral("data", self)
        self.vmTable = moose.Table("vmTable", self.data)
        self.vmTable.stepMode = 3
        self.vmTable.connect("inputRequest", self.squidCompartment, "Vm")
        self.gNaTable = moose.Table("gNaTable", self.data)
        self.gNaTable.stepMode = 3
        self.gNaTable.connect("inputRequest", self.squidCompartment.Na(), "Gk")
        self.gKTable = moose.Table("gKTable", self.data)
        self.gKTable.stepMode = 3
        self.gKTable.connect("inputRequest", self.squidCompartment.K(), "Gk")
        self.iNaTable = moose.Table("iNaTable", self.data)
        self.iNaTable.stepMode = 3
        self.iNaTable.connect("inputRequest", self.squidCompartment.Na(), "Ik")
        self.iKTable = moose.Table("iKTable", self.data)
        self.iKTable.stepMode = 3
        self.iKTable.connect("inputRequest", self.squidCompartment.K(), "Ik")
        self.nParamTable = moose.Table("nParamTable", self.data)
        self.nParamTable.stepMode = 3
        self.nParamTable.connect("inputRequest", self.squidCompartment.K(), "X")
        self.mParamTable = moose.Table("mParamTable", self.data)
        self.mParamTable.stepMode = 3
        self.mParamTable.connect("inputRequest", self.squidCompartment.Na(), "X")
        self.hParamTable = moose.Table("hParamTable", self.data)
        self.hParamTable.stepMode = 3
        self.hParamTable.connect("inputRequest", self.squidCompartment.Na(), "Y")
        self.iInjectTable = moose.Table("iInjectTable", self.data)
        self.iInjectTable.stepMode = 3
        self.iInjectTable.connect("inputRequest", self.squidCompartment, "inject") # read what is received by compartment
        self._setupClocks()
        self.getContext().useClock(0, self.squidCompartment.path+","+self.squidCompartment.path+"##")
        self.squidCompartment.useClock(1, "init")
        self.getContext().useClock(2,self.data.path+"/#")
        self.getContext().useClock(3, self.currentInjection.path)
        self._runTime = 0.050
        
    def _setupClocks(self, simDt=None, plotDt=None):
        if simDt is None:
            simDt = 1e-6
        if plotDt is None:
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
        self.getContext().setClock(3, 1e-3, 0)
        if not hasattr(self, "_clockTick2"):
            self._clockTick3 = moose.ClockTick("/sched/cj/t3")

    def save_all_plots(self):
        self.vmTable.dumpFile("vm.plot")
        print "SquidModel.dumpPlotData() - Vm is in vm.plot"
        self.iInjectTable.dumpFile("inject.plot")
        print "SquidModel.dumpPlotData() - Injection current is in squidModelIInject.plot"
        self.iNaTable.dumpFile("iNa.plot")
        print "SquidModel.dumpPlotData() - INa is in iNa.plot"
        self.iKTable.dumpFile("iK.plot")
        print "SquidModel.dumpPlotData() - IK is in iK.plot"
        self.gNaTable.dumpFile("gNa.plot")
        print "SquidModel.dumpPlotData() - GNa is in gNa.plot"
        self.gKTable.dumpFile("gK.plot")
        print "SquidModel.dumpPlotData() - GK is in gK.plot"
        
    def run(self, runTime=None):
        if runTime is not None:
            self._runTime = runTime
        self.getContext().step(self._runTime)

from pylab import *
if __name__ == "__main__":
    test_model = NoisySquid("/test")
    test_model.getContext().reset()
    test_model.run(0.1)
    plot(array(test_model.vmTable))
    show()
    test_model.save_all_plots()
    
