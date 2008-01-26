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
import numpy
import pdb
import squid
SIMDT = 1e-5
    
class InputPanel(QtGui.QFrame):
    def __init__(self, *args):
        apply(QtGui.QFrame.__init__, (self,)+args)
        self.keys = ["runtime", "prepulse", "pulsewidth", "amplitude", "updateT", "overlay", "blockNa", "blockK"]
        self.labels = {}
        self.labels["runtime"] = QtGui.QLabel("Run Time (s)")
        self.labels["prepulse"] = QtGui.QLabel("Pre-pulse time (s)")
        self.labels["pulsewidth"] = QtGui.QLabel("Pulse width (s)")
        self.labels["amplitude"] = QtGui.QLabel("Injection current (A)")
        self.labels["overlay"] = QtGui.QLabel("Overlay plots")
        self.labels["blockNa"] = QtGui.QLabel("Block Na Channels")
        self.labels["blockK"] = QtGui.QLabel("Block K Channels")
        self.labels["updateT"] = QtGui.QLabel("Graph update interval (s)")
        self.inputs = {}
        self.inputs["runtime"] = QtGui.QLineEdit("0.050")
        self.inputs["prepulse"] = QtGui.QLineEdit("0.005")
        self.inputs["pulsewidth"] = QtGui.QLineEdit("0.040")
        self.inputs["amplitude"] = QtGui.QLineEdit("1.0e-7")
        self.inputs["updateT"] = QtGui.QLineEdit("0.050")
        self.inputs["overlay"] = QtGui.QCheckBox()
        self.inputs["blockNa"] = QtGui.QCheckBox()
        self.inputs["blockK"] = QtGui.QCheckBox()
        self.setLayout(QtGui.QGridLayout(self))
        i = 0
        for key in self.keys:
            print key
            
            self.labels[key].setFrameStyle(QtGui.QFrame.Panel ^ QtGui.QFrame.Raised)
            self.layout().addWidget(self.labels[key], i, 0)
            self.layout().addWidget(self.inputs[key] , i, 1)
            i = i + 1
class PlotData():
    def __init__(self):
        self.plotData = []
        self.xEnd = 0.0

    def append(self, x=0, y=0):
        if (x == 0) or ( y == 0 ):
            self.plotData.append([])
        else:
            self.plotData.append([x,y])
        
    def concatData(self, x, y):
        if len(self.plotData) == 0  :
            self.plotData.append([x,y])
        elif len(self.plotData[-1]) != 2:
            self.plotData[-1] = [x,y]
        else:
            self.plotData[-1][0] = numpy.concatenate((self.plotData[-1][0], x))
            self.plotData[-1][1] = numpy.concatenate((self.plotData[-1][1], y))

    def lastX(self):
        if len(self.plotData) == 0 :
            print "Error: data list is empty"
        elif len(self.plotData[-1]) != 2:
            print "Error: last data entry is empty"
        else:
            return self.plotData[-1][0]

    def lastY(self):
        if len(self.plotData) == 0 :
            print "Error: data list is empty"
        elif len(self.plotData[-1]) != 2:
            print "Error: last data entry is empty"
        else:
            return self.plotData[-1][1]

    def clear(self):
        del self.plotData
        self.plotData = []
        self.append()

class Dashboard(QtGui.QFrame):
    def __init__(self, *args):
        apply(QtGui.QFrame.__init__, (self,)+args)
        self.colors = ( QtCore.Qt.white, QtCore.Qt.lightGray, QtCore.Qt.yellow, QtCore.Qt.cyan, QtCore.Qt.red, QtCore.Qt.green, QtCore.Qt.blue, QtCore.Qt.magenta, QtCore.Qt.gray )
        self.setLayout(QtGui.QHBoxLayout(self))
        # There is somthing to note here - qwt.QwtPlot is more
        # accessible, we can set the curve and the QwtPlotCurve we can
        # change the data each time. But here zooming is not
        # allowed. On the other hand qplt.Plot provides zooming, but
        # creates an ugly grid and qplt.Curve does not allow changing
        # data without recreating the whole curve object.
        self.vmGraph = qwt.QwtPlot()  
        self.vmGraph.setCanvasBackground(QtGui.QColor(0, 0, 150))
        self.curves = []
        self.addCurve("Vm")
        self.layout().addWidget(self.vmGraph)
        self.vmGraph.replot()
        self.zoomer = qwt.QwtPlotZoomer(qwt.QwtPlot.xBottom,
                                        qwt.QwtPlot.yLeft,
                                        qwt.QwtPicker.DragSelection,
                                        qwt.QwtPicker.AlwaysOn,
                                        self.vmGraph.canvas())
        self.zoomer.setRubberBandPen(QtGui.QPen(QtCore.Qt.white))
        self.zoomer.setTrackerPen(QtGui.QPen(QtCore.Qt.cyan))

    def addCurve(self, title):
        self.curves.append(qwt.QwtPlotCurve(title))
        self.curves[-1].attach(self.vmGraph)        
        self.curves[-1].setPen(QtGui.QPen(self.colors[len(self.curves)%len(self.colors)]))
        self.curves[-1].setStyle(qwt.QwtPlotCurve.Lines)

    def clear(self):
        print "Clearing dashborad"
        del self.curves
        self.vmGraph.clear()
        self.curves = []
        self.addCurve("Vm")
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
        self.connect(self.inputPanel.inputs["blockNa"], QtCore.SIGNAL("stateChanged(int)"), self.squid.toggleNaChannels)
        self.connect(self.inputPanel.inputs["blockK"], QtCore.SIGNAL("stateChanged(int)"), self.squid.toggleKChannels)
        self.layout().setMenuBar(self.menu)
        self.mainLayout.addWidget(self.dashboard)
        self.mainLayout.addWidget(self.inputPanel)
        self.mainWidget.setLayout(self.mainLayout)
        self.data = PlotData()

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
        totalTimeStr = self.inputPanel.inputs["runtime"].text()
        preTimeStr = self.inputPanel.inputs["prepulse"].text()
        pulseWidthStr = self.inputPanel.inputs["pulsewidth"].text()
        updateIntervalStr = self.inputPanel.inputs["updateT"].text()
        injectHighStr = self.inputPanel.inputs["amplitude"].text()
        injectBaseStr = "0"
        print "Read from text box", totalTimeStr, preTimeStr, pulseWidthStr, injectHighStr
        try:
            totalTime, preTime, pulseWidth, updateInterval, injectBase, injectHigh =  map(float, [totalTimeStr, preTimeStr, pulseWidthStr, updateIntervalStr, injectBaseStr, injectHighStr] )
        except Exception, vError:
            print vError
            totalTime, preTime, pulseWidth, updateInterval, injectBase, injectHigh = 0.050, 0.005, 0.040, 0.050, 0, squid.INJECT
       
        print totalTime, preTime, pulseWidth, updateInterval, injectBase, injectHigh
        import time
        t1 = time.clock()
        Vm = self.squid.doRun(totalTime, preTime, pulseWidth, injectBase, injectHigh)
        t2 = time.clock()
        print "Runtime = ", t2 - t1, "s"
        t = self.data.xEnd
        self.data.xEnd = t + totalTime
        x = numpy.arange(t, self. data.xEnd, SIMDT)
        y = numpy.array(Vm)
        self.data.concatData(x,y)
        self.dashboard.curves[-1].setData(self.data.lastX(), self.data.lastY())
        self.dashboard.vmGraph.replot() 
        self.dashboard.zoomer.setZoomBase()
        
    def slotReset(self):
        print 'Resetting the simulation'
        if self.inputPanel.inputs["overlay"].checkState() == QtCore.Qt.Checked:
            print "Overlay is on"
            self.data.append()
            self.dashboard.curves.append(qwt.QwtPlotCurve("Vm"))
            self.dashboard.addCurve("Vm")
        else:
            self.data.clear()
            self.dashboard.clear()
        
        self.dashboard.curves[-1].setData([],[])
        self.dashboard.vmGraph.replot()
       
        self.squid.getContext().reset()
        self.data.xEnd = 0.0
        


def main(args):
    app = QtGui.QApplication(sys.argv)
    mainWindow = SquidDemo()
    mainWindow.show()
    app.connect(app, QtCore.SIGNAL("lastWindowClosed()"), app, QtCore.SLOT("quit()"))
    app.exec_()

if __name__ == '__main__':
    main(sys.argv)

