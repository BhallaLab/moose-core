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
    
class InputPanel(QtGui.QFrame):
    def __init__(self, *args):
        apply(QtGui.QFrame.__init__, (self,)+args)
        self.labels = {}
        self.labels["runtime"] = QtGui.QLabel("Run Time (s)")
        self.labels["prepulse"] = QtGui.QLabel("Pre-pulse time (s)")
        self.labels["pulsewidth"] = QtGui.QLabel("Pulse width (s)")
        self.labels["amplitude"] = QtGui.QLabel("Injection current (A)")
        self.textInputs = {}
        self.textInputs["runtime"] = QtGui.QLineEdit("0.050")
        self.textInputs["prepulse"] = QtGui.QLineEdit("0.005")
        self.textInputs["pulsewidth"] = QtGui.QLineEdit("0.040")
        self.textInputs["amplitude"] = QtGui.QLineEdit("1.0e-7")

        self.setLayout(QtGui.QGridLayout(self))
        i = 0
        for key, value in self.labels.iteritems():
            value.setFrameStyle(QtGui.QFrame.Panel ^ QtGui.QFrame.Raised)
            self.layout().addWidget(value, i, 0)
            self.layout().addWidget(self.textInputs[key] , i, 1)
            i = i + 1

class Dashboard(QtGui.QFrame):
    def __init__(self, *args):
        apply(QtGui.QFrame.__init__, (self,)+args)
        self.setLayout(QtGui.QHBoxLayout(self))
        # There is somthing to note here - qwt.QwtPlot is more
        # accessible, we can set the curve and the QwtPlotCurve we can
        # change the data each time. But here zooming is not
        # allowed. On the other hand qplt.Plot provides zooming, but
        # creates an ugly grid and qplt.Curve does not allow changing
        # data without recreating the whole curve object.
        self.vmGraph = qwt.QwtPlot()  
        self.vmGraph.setCanvasBackground(QtGui.QColor(0, 0, 150))
        self.curve = qwt.QwtPlotCurve("Vm")
        self.curve.attach(self.vmGraph)        
        self.curve.setPen(QtGui.QPen(QtGui.QColor(250, 250, 0)))
        self.curve.setStyle(qwt.QwtPlotCurve.Lines)
        
        self.layout().addWidget(self.vmGraph)
        self.vmGraph.replot()


class SquidDemo(QtGui.QMainWindow):
    def __init__(self, *args):
        apply(QtGui.QMainWindow.__init__, (self,)+args)
        self.setWindowTitle('MOOSE Squid Demo')
        self.mainWidget = QtGui.QWidget()
        self.setCentralWidget(self.mainWidget)
        self.mainLayout = QtGui.QVBoxLayout(self)

        self.squid = squid.Squid('/squid')
        self.resize(600,400)

        self.initActions()
        self.initMenuBar()
        self.initToolBar()
        self.statusBar()

        self.dashboard = Dashboard()
        self.inputPanel = InputPanel()

        self.layout().setMenuBar(self.menu)
        self.mainLayout.addWidget(self.dashboard)
        self.mainLayout.addWidget(self.inputPanel)
        self.mainWidget.setLayout(self.mainLayout)

    def initActions(self):
        self.actions = {}
        # Create the actions and insert into dictionary
        self.actions["exit"] =  QtGui.QAction("Exit", self)
        self.actions["run"] = QtGui.QAction("Run", self)
        self.actions["reset"] = QtGui.QAction("Reset", self)
        # Set key-board shortcuts
        self.actions["exit"].setShortcut("Ctrl+Q")
        self.actions["run"].setShortcut("Ctrl+X")
        self.actions["reset"].setShortcut("Ctrl+R")
        # Set tool-tip text
        self.actions["exit"].setStatusTip("Exit application")   
        self.actions["run"].setStatusTip("Run the squid demo")     
        self.actions["reset"].setStatusTip("Reset the simulation")
        # Connect the actions to correct slots
        self.connect(self.actions["exit"], QtCore.SIGNAL("triggered()"), QtCore.SLOT("close()"))
        self.connect(self.actions["run"], QtCore.SIGNAL("triggered()"), self.slotRun)
        self.connect(self.actions["reset"], QtCore.SIGNAL("triggered()"), self.slotReset)

    def initMenuBar(self):
        self.menu = QtGui.QMenuBar()
        file = self.menu.addMenu("&File")
        file.addAction(self.actions["exit"])
        file.addAction(self.actions["run"])
        file.addAction(self.actions["reset"])

    def initToolBar(self):        
        self.toolbar = self.addToolBar("Exit")
        self.toolbar.addAction(self.actions["exit"])
        self.toolbar = self.addToolBar("Run")
        self.toolbar.addAction(self.actions["run"])
        self.toolbar = self.addToolBar("Reset")
        self.toolbar.addAction(self.actions["reset"])

    
    def slotRun(self):
        print 'Executing simulation'
        totalTimeStr = self.inputPanel.textInputs["runtime"].text()
        preTimeStr = self.inputPanel.textInputs["prepulse"].text()
        pulseWidthStr = self.inputPanel.textInputs["pulsewidth"].text()
        injectHighStr = self.inputPanel.textInputs["amplitude"].text()
        injectBaseStr = "0"
        print "Read from text box", totalTimeStr, preTimeStr, pulseWidthStr, injectHighStr
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
        self.dashboard.curve.setData(x,y)
        self.dashboard.zoomer = qwt.QwtPlotZoomer(qwt.QwtPlot.xBottom,
                                        qwt.QwtPlot.yLeft,
                                        qwt.QwtPicker.DragSelection,
                                        qwt.QwtPicker.AlwaysOn,
                                        self.dashboard.vmGraph.canvas())
        self.dashboard.zoomer.setRubberBandPen(QtGui.QPen(QtCore.Qt.white))
        self.dashboard.zoomer.setTrackerPen(QtGui.QPen(QtCore.Qt.cyan))

        self.dashboard.zoomer.setZoomBase()
        self.dashboard.vmGraph.replot()
        
    def slotReset(self):
        print 'Resetting the simulation'
        self.dashboard.curve.setData([],[])
        self.dashboard.vmGraph.replot()
        self.squid.getContext().reset()
        


def main(args):
    app = QtGui.QApplication(sys.argv)
    mainWindow = SquidDemo()
    mainWindow.show()
    app.connect(app, QtCore.SIGNAL("lastWindowClosed()"), app, QtCore.SLOT("quit()"))
    app.exec_()

if __name__ == '__main__':
    main(sys.argv)

