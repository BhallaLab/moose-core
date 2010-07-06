# mooseplots.py --- 
# 
# Filename: mooseplots.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Mon Jul  5 21:35:09 2010 (+0530)
# Version: 
# Last-Updated: Tue Jul  6 21:09:33 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 298
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Class to handle plotting in MOOSE GUI
# 
# 

# Change log:
# 
# 
# 
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 
# 

# Code:


from PyQt4 import QtGui
from PyQt4.Qt import Qt
import PyQt4.Qwt5 as Qwt
from PyQt4.Qwt5.anynumpy import *

import moose

class MoosePlot(Qwt.QwtPlot):
    """Handler for plots in MOOSE gui"""
    plot_index = 0
    colors = [ Qt.red, 
               Qt.blue, 
               Qt.darkYellow, 
               Qt.green, 
               Qt.magenta, 
               Qt.darkCyan,
               Qt.black, 
               Qt.cyan, 
               Qt.darkRed, 
               Qt.darkGreen, 
               Qt.yellow, 
               Qt.darkMagenta,  
               Qt.gray,   
               Qt.darkBlue, 
               Qt.lightGray ]
    
    def __init__(self, *args):
        Qwt.QwtPlot.__init__(self, *args)
        print 'In __init__'
        self.plotNo = MoosePlot.plot_index
        self.steptime = 1e-3
        self.simtime = 100e-3
        MoosePlot.plot_index = MoosePlot.plot_index + 1
        self.curveIndex = 0
        self.setCanvasBackground(Qt.white)
        self.alignScales()
        self.xmin = 100.1
        self.curveTableMap = {} # curve -> moose table
        self.tableCurveMap = {} # moose table -> curve
        # title = Qwt.QwtText('Plot %d' % (self.plotNo), Qwt.QwtText.PlainText)        
        # font = QtGui.QFont('Helvetica', 10)
        # # font.setBold(True)
        # font.setWeight(QtGui.QFont.Bold)
        # title.setFont(font)
        # print 'setting title', title.text()
        # print ' ----- '
        # print self.title().font().pointSize()
        # self.setTitle(title)
        mY = Qwt.QwtPlotMarker()
        mY.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
        mY.setLineStyle(Qwt.QwtPlotMarker.HLine)
        mY.setYValue(0.0)
        mY.attach(self)
        ###########################################
        # This is a place holder curve
        ###########################################
        # dummyCurve = Qwt.QwtPlotCurve('Dummy curve')
        # xx = arange(0.0, 100.1, 0.5)
        # yy = sin(xx * pi / 27.18282)
        # dummyCurve.setData(xx, yy)
        # dummyCurve.setPen(MoosePlot.colors[0])
        # dummyCurve.attach(self)
        ###########################################
        xtitle = Qwt.QwtText('Time (ms)')
        ytitle = Qwt.QwtText('Value')
        if self.parent():
            xtitle.setFont(self.parent().font())
            ytitle.setFont(self.parent().font())
        else:
            xtitle.setFont(QtGui.QFont("Helvetica", 18))
            ytitle.setFont(QtGui.QFont("Helvetica", 18))
        print xtitle.font().pointSize()
        self.setAxisTitle(Qwt.QwtPlot.xBottom, xtitle)
        self.setAxisTitle(Qwt.QwtPlot.yLeft, ytitle)


    def alignScales(self):
        self.canvas().setFrameStyle(QtGui.QFrame.Box | QtGui.QFrame.Plain)
        self.canvas().setLineWidth(1)
        for ii in range(Qwt.QwtPlot.axisCnt):
            scaleWidget = self.axisWidget(ii)
            if scaleWidget:
                scaleWidget.setMargin(0)
            scaleDraw = self.axisScaleDraw(ii)
            if scaleDraw:
                scaleDraw.enableComponent(Qwt.QwtAbstractScaleDraw.Backbone, False)

    def updatePlot(self, currentTime):
        if currentTime > self.xmin * 1e-3:
            self.xmin = currentTime * 1e3
        for curve, table in self.curveTableMap.items():
            tabLen = len(table)
            print 'MoosePlot.updatePlot - table %s is of length %d.' % (table.path, tabLen)
            if tabLen == 0:
                continue
            ydata = array(table.table)
            xdata = linspace(0, currentTime, tabLen)
            curve.setData(xdata, ydata)
        print 'MoosePlot -- updatePlot: replotting'
        self.replot()

    def timerEvent(self, e):
        # print 'Timer event'
        current_time = moose.PyMooseBase.getContext().getCurrentTime()
        if current_time + self.steptime < self.simtime:
            moose.PyMooseBase.getContext().step(self.steptime)
        elif current_time < self.simtime:
            moose.PyMooseBase.getContext().step(self.simtime - current_time)
        else: 
            self.killTimer(self.timer)
        current_time = moose.PyMooseBase.getContext().getCurrentTime()
        if current_time > self.xmin * 1e-3:
            self.xmin = current_time * 1e3
        for curve, table in self.curveTableMap.items():
            if len(table) == 0:
                print 'Table is 0 length'
                continue
            else:
                print 'Table is %d length' % (len(table))
            xx = linspace(0.0, current_time * 1e3, len(table))
            yy = array(table.table)
            curve.setData(xx, yy)
            print 'timer event', max(yy)
        self.replot()

    def addTable(self, table):
        curve = Qwt.QwtPlotCurve(table.name)
        curve.setPen(MoosePlot.colors[self.curveIndex])
        self.curveIndex = (self.curveIndex + 1) % len(MoosePlot.colors)
        self.curveIndex = self.curveIndex + 1
        self.curveTableMap[curve] = table
        self.tableCurveMap[table] = curve
        current_time = moose.PyMooseBase.getContext().getCurrentTime() * 1e3
        if current_time > self.xmin:
            self.xmin = current_time
        if len(table) > 0:
            yy = array(table.table)
            xx = linspace(0.0, current_time, len(yy))
            curve.setData(xx, yy)
        curve.attach(self)
        self.replot()

    def removeTable(self, table):
        try:
            curve = self.tableCurveMap.pop(table)
            curve.detach()
            self.curveTableMap.pop(curve)
        except KeyError:
            pass

import sys
if __name__ == '__main__':
    app = QtGui.QApplication([])
    testComp = moose.Compartment('c')
    testTable = moose.Table('t')
    testTable.stepMode = 3
    testTable.connect('inputRequest', testComp, 'Vm')
    testPulse = moose.PulseGen('p')
    testPulse.firstDelay = 50e-3
    testPulse.firstWidth = 40e-3
    testPulse.firstLevel = 1e-9
    testPulse.connect('outputSrc', testComp, 'injectMsg')
    context = moose.PyMooseBase.getContext()
    simdt = 1e-4/4
    context.setClock(0, simdt)
    context.setClock(1, simdt)
    context.setClock(2, simdt)
    stop = 1000 # stop every 1000 steos
    simtime = 500e-3
    context.reset()
    plot = MoosePlot()
    plot.addTable(testTable)
    plot.show()
        
    sys.exit(app.exec_())

# 
# mooseplots.py ends here
