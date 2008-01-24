#!/usr/bin/env python

######################################################################
# File:           qtSquid.py
# Description:    This is the PyQt4 implementation of a small gui 
#                 for pymoose. 
# Author:         Subhasis Ray
# Date:           2007-12-05 14:28:37
# Copyleft:       Subhasis Ray, NCBS, 2007
# License:        GPL3
######################################################################
import sys
from PyQt4 import QtGui, QtCore
from PyQt4 import Qwt5 as qwt
import PyQt4.Qwt5.qplt as qplt
import squid
SIMDT = 1e-5
class SquidDemo(QtGui.QMainWindow):
    def __init__(self, *args):
        apply(QtGui.QMainWindow.__init__, (self,)+args)
        self.setWindowTitle('MOOSE Squid Demo')
        self.mainWidget = QtGui.QWidget(self)
        self.setCentralWidget(self.mainWidget)
        self.mainLayout = QtGui.QVBoxLayout(self.mainWidget)

        self.squid = squid.Squid('/squid')
        self.resize(600,400)

        self.menu = QtGui.QMenuBar()
        
        self.exit = QtGui.QAction('Exit', self)
        self.exit.setShortcut('Ctrl+Q')
        self.exit.setStatusTip('Exit application')        
        self.connect(self.exit, QtCore.SIGNAL('triggered()'), QtCore.SLOT('close()'))
        
        self.run = QtGui.QAction('Run', self)
        self.run.setShortcut('Ctrl+X')
        self.run.setStatusTip('Run the squid demo')
        self.connect(self.run, QtCore.SIGNAL('triggered()'), self.runDemo)
        
        self.clear = QtGui.QAction('Clear', self)
        self.clear.setShortcut('Ctrl+L')
        self.clear.setStatusTip('Clear the graph panel')
        self.connect(self.clear, QtCore.SIGNAL('triggered()'), self.clearGraph)

        self.reset = QtGui.QAction('Reset', self)
        self.reset.setShortcut('Ctrl+R')
        self.reset.setStatusTip('Reset the simulation')
        self.connect(self.reset, QtCore.SIGNAL('triggered()'), self.resetSimulation)

        self.statusBar()

        file = self.menu.addMenu('&File')
        file.addAction(self.exit)
        file.addAction(self.run)
        file.addAction(self.clear)
        file.addAction(self.reset)

        self.mainLayout.setMenuBar(self.menu)
        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(self.exit)
        self.toolbar = self.addToolBar('Run')
        self.toolbar.addAction(self.run)
        self.toolbar = self.addToolBar('Clear')
        self.toolbar.addAction(self.clear)
        self.toolbar = self.addToolBar('Reset')
        self.toolbar.addAction(self.reset)
        self.totalTime = QtGui.QLineEdit("Total Time (s)")
        self.preTime = QtGui.QLineEdit("Time before injection (s)")
        self.pulseWidth = QtGui.QLineEdit("Pulse width (s)")
        self.injectBase = QtGui.QLineEdit("Base current (A)")
        self.injectHigh = QtGui.QLineEdit("Pulse amplitude (A)")
        
        self.inputsLayout = QtGui.QHBoxLayout()
        self.inputsLayout.addWidget(self.totalTime)
        self.inputsLayout.addWidget(self.preTime)
        self.inputsLayout.addWidget(self.pulseWidth)
        self.inputsLayout.addWidget(self.injectBase)
        self.inputsLayout.addWidget(self.injectHigh)
        self.plotLayout = QtGui.QVBoxLayout()
        self.graph = qplt.Plot(self)
        self.connect(self.clear, QtCore.SIGNAL('triggered()'), self, QtCore.SLOT('clearGraph()'))
        self.plotLayout.addWidget(self.graph)
        self.mainLayout.addLayout(self.plotLayout)
        self.mainLayout.addLayout(self.inputsLayout)


    def runDemo(self):
        print 'Executing simulation'
        totalTimeStr = self.totalTime.text()
        preTimeStr = self.preTime.text()
        pulseWidthStr = self.pulseWidth.text()
        injectBaseStr = self.injectBase.text()
        injectHighStr = self.injectHigh.text()
        print  "Read from text box", totalTimeStr, preTimeStr, pulseWidthStr, injectBaseStr, injectHighStr
        try:
            totalTime, preTime, pulseWidth, injectBase, injectHigh =  map(float, [totalTimeStr, preTimeStr, pulseWidthStr, injectBaseStr, injectHighStr] )
        except Exception, vError:
            print vError
            totalTime, preTime, pulseWidth, injectBase, injectHigh = 0.050, 0.005, 0.040, 0, squid.INJECT
       
        print totalTime, preTime, pulseWidth, injectBase, injectHigh
        Vm = self.squid.doRun(totalTime, preTime, pulseWidth, injectBase, injectHigh)

        t = 0.0
        y = qplt.array(Vm)
        x =  qplt.zeros(len(y))
        for i in range(len(x)):
            x[i] = t
            t = t + SIMDT
        self.graph.plotCurve(qplt.Curve(x,y))

    def clearGraph(self):
        print 'Clearing graph'
        self.graph.clear()
        self.graph.replot()
        
    def resetSimulation(self):
        print 'Resetting the simulation'
        self.graph.clear()
        self.graph.replot()
        self.squid.getContext().reset()
        


def main(args):
    app = QtGui.QApplication(sys.argv)
    mainWindow = SquidDemo()
    mainWindow.show()
    app.connect(app, QtCore.SIGNAL("lastWindowClosed()"), app, QtCore.SLOT("quit()"))
    app.exec_()

if __name__ == '__main__':
    main(sys.argv)

