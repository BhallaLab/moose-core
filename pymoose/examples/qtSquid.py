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
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.resize(600,400)
        self.setWindowTitle('MOOSE Squid Demo')
        menuBar = self.menuBar()
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
        self.connect(self.clear, QtCore.SIGNAL('triggered()'), self.clearPlot)
        
        self.statusBar()
        file = menuBar.addMenu('&File')
        file.addAction(self.exit)
        file.addAction(self.run)
        file.addAction(self.clear)
        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(self.exit)
        self.toolbar = self.addToolBar('Run')
        self.toolbar.addAction(self.run)
        self.toolbar = self.addToolBar('Clear')
        self.toolbar.addAction(self.clear)
        
        self.graph = qplt.Plot(self)
        self.setCentralWidget(self.graph)

    def runDemo(self):
        print 'Executing simulation'
        Vm = squid.runDemo()
        t = 0.0
        y = qplt.array(Vm)
        x =  qplt.zeros(len(y))
        for i in range(len(x)):
            x[i] = t
            t = t + SIMDT
        self.graph.plotCurve(qplt.Curve(x,y))
        
    def clearPlot(self):
        print 'Clearing the plot'
        self.graph = qplt.Plot(self)
        self.setCentralWidget(self.graph)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    mainWindow = SquidDemo()
    mainWindow.show()
    app.exec_()

