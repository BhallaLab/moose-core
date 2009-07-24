# mooseplots.py --- 
# 
# Filename: mooseplots.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Wed Jul  1 12:58:00 2009 (+0530)
# Version: 
# Last-Updated: Fri Jul 24 13:37:46 2009 (+0530)
#           By: subhasis ray
#     Update #: 317
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
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
import math
import sys
sys.path.append('..')
from collections import defaultdict

from PyQt4 import QtGui, QtCore
from PyQt4 import Qwt5 as Qwt
import numpy

from filetypeutil import FileTypeChecker
import moose

class MoosePlots(QtGui.QTableWidget):
    colors = [ QtCore.Qt.red, 
               QtCore.Qt.blue, 
               QtCore.Qt.darkYellow, 
               QtCore.Qt.green, 
               QtCore.Qt.magenta, 
               QtCore.Qt.darkCyan,
               QtCore.Qt.black, 
               QtCore.Qt.cyan, 
               QtCore.Qt.darkRed, 
               QtCore.Qt.darkGreen, 
               QtCore.Qt.yellow, 
               QtCore.Qt.darkMagenta,  
               QtCore.Qt.gray,   
               QtCore.Qt.darkBlue, 
               QtCore.Qt.lightGray ]

    """Container for plots in MOOSE."""
    def __init__(self, *args):
	QtGui.QTableWidget.__init__(self, *args)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.plots = []
	self.plot_data_map = defaultdict(set)
        self.data_plot_map = {}        
        self.container_table_map = defaultdict(set) # container name - table map
        self.table_curve_map = {} # This is just for holding the reference of the curve

    def loadPlots(self, mooseHandler, filetype, mooseContainers=None):
        self.plots = []
        self.plot_data_map.clear()
        self.container_table_map.clear()
        self.table_curve_map.clear() # is just for holding the reference of the curve
        self.clear()
        if mooseContainers is None:
            mooseContainers = []
            if filetype == FileTypeChecker.type_kkit:
                mooseContainers.append(moose.Neutral('/graphs'))
                mooseContainers.append(moose.Neutral('/moregraphs'))
            elif (filetype == FileTypeChecker.type_genesis) or (filetype == FileTypeChecker.type_sbml):
                mooseContainers.append(moose.Neutral('/data'))
            else:
                print "File Type:", filetype, "- Don't know where to look for plot data"
                
            
        for container in mooseContainers:
            for child_id in container.children():
                child_obj = moose.Neutral(child_id)
                if child_obj.className == 'Neutral':                        
                    for table_id in child_obj.children():
                        table = moose.Table(table_id)
                        if table.className != 'Table':
                            continue
                        else:
                            self.container_table_map[child_obj.name].add(table)
                elif child_obj.className == 'Table':
                    self.container_table_map[child_obj.name].add(moose.Table(child_id))
                else:
                    print 'Could not find any table under containers'
            
        # Now create the plots
        plot_count = len(self.container_table_map.keys())
        if plot_count is 0:
            print 'Nothing to plot'
            return
        
        rows = int(round(math.sqrt(plot_count)))
        if rows is 0:
            rows = 1
        cols = int(math.ceil(float(plot_count) / rows))
        row = 0
        col = 0
        print '#########: plots:', plot_count, ' rows:', rows, ' cols:', cols
        self.setRowCount(rows)
        self.setColumnCount(cols)
        plot = None
        for name, dataset in self.container_table_map.items():
            if len(dataset) is 0:
                self.container_table_map.pop(name)
                continue
            plot = Qwt.QwtPlot(self)
            sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
            plot.setSizePolicy(sizePolicy)
            plot.insertLegend(Qwt.QwtLegend(), Qwt.QwtPlot.BottomLegend)
            color_no = 0
            for table in dataset:
                if len(table) == 0:
                    continue
                self.plot_data_map[plot].add(table)
                curve = Qwt.QwtPlotCurve(self.tr(table.name))
                self.table_curve_map[table.path] = curve
#                     print 'Created curve for:', table.path
                ydata = numpy.array(table)            
                xdata = numpy.linspace(0, mooseHandler.currentTime(), len(table))
                curve.setData(xdata, ydata)
                curve.setPen(MoosePlots.colors[color_no])
                color_no += 1
                curve.attach(plot)
                plot.replot()
            self.setCellWidget(row, col, plot)
            col += 1
            if col >= cols:
                row += 1
                col = 0
        if plot is not None:
            width = plot.sizeHint().width()
            height = plot.sizeHint().height()
            for col in range(self.verticalHeader().count()):
                self.verticalHeader().resizeSection(col, width)
                print 'resized col:', col
            for row in range(self.horizontalHeader().count()):
                self.horizontalHeader().resizeSection(row, height)
                print 'resized row:', row
#                 self.verticalHeader().setDefaultSectionSize(width)
#                 self.horizontalHeader().setDefaultSectionSize(height)
#                 self.horizontalHeader().resizeSections()
#                 self.verticalHeader().resizeSections()
            print height, width
                
        self.update()

    def updatePlots(self, mooseHandler):        
        """Update the plots"""
        runtime = mooseHandler.runTime
        for plot, table_set in self.plot_data_map.items():
            for table in table_set:
#                 print 'updating:', table.name
                dt = mooseHandler.getDt(table)
#                 print table.name, 'dt =', dt, 'runtime =', runtime
                if dt > 0:
                    steps = math.ceil(runtime/dt)
                    xdata = numpy.linspace(0, runtime, steps)
                    ydata = numpy.array(table)
#                     print 'length of table:', len(table), 'length of xdata:', len(xdata)
                    plot.setAxisScale(Qwt.QwtPlot.xBottom, 0, runtime, runtime / 5.0)
                    curve = self.table_curve_map[table.path]
                    curve.setData(xdata, ydata)
            plot.replot()
            plot.show()
        self.update()

    def rescalePlots(self):
        for plot in self.plot_data_map.keys():
            plot.setAxisAutoScale(Qwt.QwtPlot.xBottom)
            plot.replot()
        self.update()


    def resetPlots(self):
        for plot, table_set in self.plot_data_map.items():
            for table in table_set:
                curve = self.table_curve_map[table.path]
                curve.setData([], [])
                plot.replot()
        self.update()
        

# 
# mooseplots.py ends here
