#/*******************************************************************
# * File:            axon_v1.py
# * Description:      
# * Author:          Subhasis Ray
# * E-mail:          ray dot subhasis at gmail dot com
# * Created:         2008-10-07 19:19:03
# ********************************************************************/
#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/
import sys
sys.path.append("../channels")
sys.path.append("../..")
from bulbchan import *

import moose
import bulbchan

SIMDT = 50e-6
IODT = 50e-6
SIMLENGTH = 0.05
INJECT = 5e-10
EREST_ACT = -0.065

def printTree(node, indent="", maxdepth=1, curdepth=0):
    if curdepth >= maxdepth:
        return
    
    nn = moose.Neutral(node)
    for child in nn.children():
        print curdepth, (indent + moose.Neutral(child).name)
        printTree(child, indent + " ", maxdepth, curdepth+1)
        
class Axon():
    def __init__(self, *args):
        global SIMDT, IODT, SIMLENGTH, INJECT, EREST_ACT
        moose.PyMooseBase.getContext().setCwe("/library")
        bulbchan.make_Na_mit_usb()
        bulbchan.make_K_mit_usb()
        moose.PyMooseBase.getContext().setCwe("/")
        
        moose.PyMooseBase.getContext().readCell("axon.p", "/axon")
        print "Axon.__init__: readCell - done."
        self.model = moose.Cell("/axon")
        print "Printing model tree"
        printTree("/axon/soma")
        self.plots = moose.Neutral("/plots")
        self.vm0Table = moose.Table("/plots/Vm0")
        self.vm0Table.xdivs = int(SIMLENGTH / IODT)
        self.vm0Table.xmin = 0.0
        self.vm0Table.xmax = SIMLENGTH
        self.vm0Table.stepMode = 3
        self.soma = moose.Compartment("/axon/soma")
        self.vm0Table.connect("inputRequest", self.soma, "Vm")

        # Setup table to simulate time course of current injection
        self.injectTable = moose.Table("/inject")
        self.injectTable.xdivs = 100
        self.injectTable.xmin = 0.0
        self.injectTable.xmax = SIMLENGTH
        self.injectTable.stepMode = 2
        self.injectTable.stepSize = 0.0
        self.injectTable.connect("outputSrc", self.soma, "injectMsg")
        current = INJECT
        for i in range(101):
            if i % 20 == 0:
                current = INJECT - current
                self.injectTable[i] = current

        self.injectPlot = moose.Table("/plots/inject")
        self.injectPlot.stepMode = 3
        self.injectPlot.connect("inputRequest", self.soma, "inject")

        self.iNaPlot = moose.Table("/plots/INa")
        self.iNaPlot.stepMode = 3
        self.iNaPlot.connect("inputRequest", moose.HHChannel("/axon/soma/Na_mit_usb"), "Ik")

        self.iKPlot = moose.Table("/plots/IK")
        self.iKPlot.stepMode = 3
        self.iKPlot.connect("inputRequest", moose.HHChannel("/axon/soma/K_mit_usb"), "Ik")
        
        moose.PyMooseBase.getContext().setClock(0, SIMDT, 0)
        moose.PyMooseBase.getContext().setClock(1, SIMDT, 0)
        moose.PyMooseBase.getContext().setClock(2, IODT, 0)
        self.injectTable.useClock(0)
 #        moose.PyMooseBase.getContext().useClock(0, "/axon,/axon/##")
        moose.PyMooseBase.getContext().useClock(2, "/plots/#[TYPE=table]")

        print "Axon.__init__ - end"

    def simulate(self):
        global SIMDT, IODT, SIMLENGTH, INJECT, EREST_ACT
        # for testing
#        self.model.method = 'ee'
        # !for testing
        moose.PyMooseBase.getContext().reset()
        moose.PyMooseBase.getContext().step(SIMLENGTH)
        
    def savePlots(self):
        plots = moose.Neutral("/plots")
        for child in plots.children():
            table = moose.Table(child)
            fileName = table.name + ".plot"
            table.dumpFile(fileName)
            print "Data saved in", fileName

if __name__ == "__main__":
    axon = Axon()
    axon.simulate()
    axon.savePlots()
    print "main - end"
