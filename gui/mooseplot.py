# mooseplots.py --- 
# 
# Filename: mooseplots.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Mon Jul  5 21:35:09 2010 (+0530)
# Version: 
# Last-Updated: Wed Nov  2 16:56:07 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 724
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
from PyQt4 import QtCore
import PyQt4.Qwt5 as Qwt
from PyQt4.Qwt5.anynumpy import *

import config
import moose
import os

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
        self.plotNo = MoosePlot.plot_index
        MoosePlot.plot_index = MoosePlot.plot_index + 1
        self.runcount = 0
        self.setAcceptDrops(True)
        self.curveIndex = 0
        self.setCanvasBackground(Qt.white)
        self.alignScales()
        self.xmin = 100.1
        self.curveTableMap = {} # curve -> moose table
        self.tableCurveMap = {} # moose table -> curve
        self.overlay = False
        legend = Qwt.QwtLegend()
        legend.setItemMode(Qwt.QwtLegend.ClickableItem)#CheckableItem)
        self.insertLegend(legend, Qwt.QwtPlot.BottomLegend)
        # self.setTitle('Plot %d' % (self.plotNo))
        mY = Qwt.QwtPlotMarker()
        mY.setLabelAlignment(Qt.AlignRight | Qt.AlignTop)
        mY.setLineStyle(Qwt.QwtPlotMarker.HLine)
        mY.setYValue(0.0)
        mY.attach(self)
        xtitle = Qwt.QwtText('Time (s)')
        ytitle = Qwt.QwtText('Value')
        if self.parent():
            xtitle.setFont(self.parent().font())
            ytitle.setFont(self.parent().font())
        else:
            xtitle.setFont(QtGui.QFont("Helvetica", 18))
            ytitle.setFont(QtGui.QFont("Helvetica", 18))
        self.setAxisTitle(Qwt.QwtPlot.xBottom, xtitle)
        self.setAxisTitle(Qwt.QwtPlot.yLeft, ytitle)
        self.zoomer = Qwt.QwtPlotZoomer(Qwt.QwtPlot.xBottom,
                                   Qwt.QwtPlot.yLeft,
                                   Qwt.QwtPicker.DragSelection,
                                   Qwt.QwtPicker.AlwaysOn,
                                   self.canvas())
        self.zoomer.setRubberBandPen(QtGui.QPen(Qt.black))
        self.zoomer.setTrackerPen(QtGui.QPen(Qt.black))
        self.mooseHandler = None
        #add_chait, commented below, calling same from moosegui.py
	#QtCore.QObject.connect(self, QtCore.SIGNAL("legendChecked(QwtPlotItem *,bool)"), self.plotItemClicked)

    def clearZoomStack(self):
        """Auto scale and clear the zoom stack
        """
        self.setAxisAutoScale(Qwt.QwtPlot.xBottom)
        self.setAxisAutoScale(Qwt.QwtPlot.yLeft)
        self.replot()
        self.zoomer.setZoomBase()

    def reconfigureSelectedCurves(self, pen, symbol, style, attribute,newLegendName,curve): 
        #add_chait, newLegendName,curve (qwtlegentitem)
        """Reconfigure the selected curves to use pen for line and
        symbol for marking the data points."""
        print 'Reconfiguring slected plots'
        for item in self.itemList():
            widget = self.legend().find(item)
            if isinstance(widget, Qwt.QwtLegendItem) and curve == item: #add_chait,quickfix to edit just the selected curve
                item.setPen(pen)
                item.setSymbol(symbol)
                item.setStyle(style)
                item.setCurveAttribute(attribute)
                item.setTitle(newLegendName)
        self.replot()

    def fitSelectedPlots(self):
        for item in self.itemList():
            widget = self.legend().find(item)
            if isinstance(widget, Qwt.QwtLegendItem) and widget.isChecked():
                item.setCurveAttribute(Qwt.QwtPlotCurve.Fitted)
        self.replot()
            
    def showSelectedCurves(self, on):
        for item in self.itemList():
            widget = self.legend().find(item)
            if isinstance(widget, Qwt.QwtLegendItem) and widget.isChecked():
                item.setVisible(on)
        self.replot()

    def showAllCurves(self):
        for item in self.itemList():
            if isinstance(item, Qwt.QwtPlotCurve):
                print item
                item.setVisible(True)
        self.replot()

    def setLineStyleSelectedCurves(self, style=Qwt.QwtPlotCurve.NoCurve):        
        for item in self.itemList():
            widget = self.legend().find(item)
            if isinstance(widget, Qwt.QwtLegendItem) and widget.isChecked():
                item.setStyle(style)
        self.replot()

    def setSymbol(self, 
                       symbolStyle=None, 
                       brushColor=None, brushStyle=None, 
                       penColor=None, penWidth=None, penStyle=None, 
                       symbolHeight=None, symbolWidth=None):
        """Set the symbol used in plotting.

        This function gives overly flexible access to set the symbol
        of all the properties of the currently selected curves. If any
        parameter is left unspecified, the existing value of that
        property of the symbol is maintained.

        TODO: create a little plot-configuration widget to manipulate each
        property of the selected curves visually. That should replace setting setSymbol amd setLineStyle.

        """
        for item in self.itemList():
            widget = self.legend().find(item)
            if isinstance(widget, Qwt.QwtLegendItem) and widget.isChecked():
                oldSymbol = item.symbol()
                if symbolStyle is None:
                    symbolStyle = oldSymbol.style()
                if brushColor is None:
                    brushColor = oldSymbol.brush().color()
                if brushStyle is None:
                    brushStyle = oldSymbol.brush().style()
                if penColor is None:
                    penColor = oldSymbol.pen().color()
                if penWidth is None:
                    penWidth = oldSymbol.pen().width()
                if penStyle is None:
                    penStyle = oldSymbol.pen().style()
                if symbolHeight is None:
                    symbolHeight = oldSymbol.size().height()
                if symbolWidth is None:
                    symbolWidth = oldSymbol.size().width()
                pen = QtGui.QPen(penColor, penWidth, penStyle)
                symbol = Qwt.QwtSymbol(symbolStyle, oldSymbol.brush(), pen, QtCore.QSize(width, height)) 
                item.setSymbol(symbol)
        self.replot()

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
        self.xmin = currentTime
        for curve, table in self.curveTableMap.items():
            tabLen = len(table)
            if tabLen < 2:
                continue
            ydata = array(table)           
            xdata = linspace(0, currentTime, tabLen)
            #~ harsha:for Genesis first element had some invalid number which when ploted had a different result so eliminating
            #~ curve.setData(xdata, ydata)            
            curve.setData(xdata[2:tabLen:1],ydata[2:tabLen:1])
        self.replot()
        self.clearZoomStack()
        self.zoomer.setZoomBase()
        

    def addTable(self, table): 
        new_curve = None
        curve = None
        curve_name = table.name
        try:
            if self.overlay:
                curve = self.tableCurveMap.pop(table)                
                curve.setTitle('%s # %d' % (table.name, self.runcount))
                print 'Setting old curve name to:', curve.title()
                new_curve = Qwt.QwtPlotCurve(curve_name)
            else:
                curve = self.tableCurveMap[table]
        except KeyError:
            print 'Adding table ', table.path
            new_curve = Qwt.QwtPlotCurve(curve_name)
        if new_curve:
            new_curve.setPen(MoosePlot.colors[self.curveIndex])
            self.curveIndex = (self.curveIndex + 1) % len(MoosePlot.colors)
            self.curveTableMap[new_curve] = table
            self.tableCurveMap[table] = new_curve
            new_curve.attach(self)
        if len(table) > 2:
            yy = array(table)
            xx = linspace(0.0, self.xmin, len(yy))
            if new_curve:
                new_curve.setData(xx, yy)
            else:
                curve.setData(xx, yy)

    def removeTable(self, table):
        try:
            curve = self.tableCurveMap.pop(table)
            curve.detach()
            self.curveTableMap.pop(curve)
        except KeyError:
            pass

    def setOverlay(self, overlay):
        self.overlay = overlay

    def reset(self, runcount):
        print 'Resetting curves'
        self.runcount = runcount
        tables = self.tableCurveMap.keys()
        if not self.overlay:
            self.clear()
        else:
            for curve, table in self.curveTableMap.items():
                title = '%s # %d' % (table.name, self.runcount)
                curve.setTitle(title)
        self.tableCurveMap.clear()
        self.curveTableMap.clear()
        for table in tables:
            self.addTable(table)

    def detachItems(self):
        self.tableCurveMap.clear()
        self.curveTableMap.clear()
        QtGui.QwtPlotDict.detachItems(self)

    def plotItemClicked(self,item,value):
	if(item.isVisible):
		''' Initially all the item.isVisible is true'''
		item.setVisible(not item.isVisible)
                item.isVisible = False
		item.setItemAttribute(Qwt.QwtPlotItem.AutoScale,False);	                 
        else:
                '''If the item.isVisible is made false (say hidden) here makes true'''
                item.setVisible(not item.isVisible)
                item.isVisible = True
                item.setItemAttribute(Qwt.QwtPlotItem.AutoScale,True);	                 
            
	self.replot()
 		

    def dragEnterEvent(self, event):        
        event.accept()

    def dropEvent(self, event):
        """Overrides QWidget's method to accept drops of fields from
        ObjectEditor.

        """
        source = event.source()
        # Should check that source is objectEditor - right now we don't have
        # any other source for Plot, so don't bother.
        model = source.model()
        index = source.currentIndex()
        if index.isValid():
            # This is horrible code as I am peeping into the
            # ObjectEditor's internals, but I don't have the time or
            # patience to implement Drag objects for ObjectEditor.
            fieldName = model.fields[index.row()]
            fieldPath = model.mooseObject.path + '/' + fieldName
            # Till now this file was decoupled from MooseHandler. Now
            # I am going to break that for the sake of getting the job
            # done quick and dirty.

            # This is terrible code ... I would have been ashamed of
            # this unless it was a brainless typing session to meet
            # the immediate needs. I hope some day somebody with the
            # time and skill to do "Software Engineering" will clean
            # this mishmash of dependencies.
            # -- Subha
#edit_chait
#            table = self.mooseHandler.addFieldTable(fieldPath)
#            tokens = fieldPath.split('/')
#            if len(tokens) < 2:
#                raise IndexError('Field path should have at least two components. Got %d' % (len(tokens)))
#            self.addTable(table, tokens[-2] + '_' + tokens[-1])

            self.emit(QtCore.SIGNAL('draggedAField(const QString&,const QString&)'),fieldPath,self.objectName())
            # This also breaks the capability to move a plot from one
            # plot window to another.
            model.updatePlotField(index, self.objectName())

    def savePlotData(self, directory=''):
        for table in self.tableCurveMap.keys():
            filename = os.path.join(directory, table.name + '.plot')
            print 'Saving', filename
            table.dumpFile(filename)

class MoosePlotWindow(QtGui.QMdiSubWindow):
    """This is to customize MDI sub window for our purpose.

    In particular, we don't want anything to be deleted when the window is closed. 
    
    """
    def __init__(self, *args):
        QtGui.QMdiSubWindow.__init__(self, *args)
        
    def closeEvent(self, event):
        self.emit(QtCore.SIGNAL('subWindowClosed()'))
        self.hide()
        
   

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
