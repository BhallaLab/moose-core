# mplot.py --- 
# 
# Filename: mplot.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Mar 11 20:24:26 2013 (+0530)
# Version: 
# Last-Updated: Mon Mar 11 22:23:26 2013 (+0530)
#           By: subha
#     Update #: 210
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Moose plot widget default implementation. This should be rich enough
# to suffice for most purposes.
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
"""
    *TODO*

    1) Option for default colors, markers, etc.

    2) Option for configuring number of rows and columns of
    subplots. (I think matplotlib grids will be a bit too much to
    implement). Problem is this has to be done before actual axes are
    created (as far as I know). Idea: can we do something like movable
    widgets example in Qt?

    3) Option for selecting any line or set of lines and change its
    configuration (as in dataviz).

    4) Association between plots and the data source.

    5) Lots and lots of scipy/numpy/scikits/statsmodels utilities can be added. To
    start with, we should have 
      a)digital filters
      b) fft
      c) curve fitting
    
    6) For (5), think of another layer of plugins. Think of this as a
    standalone program. All these facilities should again be
    pluggable. We do not want to overwhelm novice users with fancy
    machine-learning stuff. They should be made available only on
    request.
    
"""


__author__ = "Subhasis Ray"

import numpy as np
from PyQt4 import QtGui, QtCore
from PyQt4.Qt import Qt
from matplotlib import mlab
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

from mplugin import PlotBase

class CanvasWidget(FigureCanvas):
    """Widget to draw plots on.

    This class keep track of all the axes in a dictionary. The key for
    an axis is its index number in sequence of creation. 

    next_id: The key for the next axis. 

    current_id: Key for current axis (anu plotting will happen on
    this).

    """
    def __init__(self, *args, **kwargs):
        self.figure = Figure()
        FigureCanvas.__init__(self, self.figure, *args, **kwargs)
        self.axes = {}
        self.next_id = 0
        self.current_id = -1

    def addSubplot(self, rows, cols):        
        """Add a subplot to figure and set it as current axes."""
        self.axes[self.next_id] = self.figure.add_subplot(rows, cols, self.next_id+1)
        self.axes[self.next_id].set_title(chr(self.next_id + ord('A')))
        self.current_id = self.next_id
        self.next_id += 1
        
    def plot(self, *args, **kwargs):
        self.callAxesFn('plot', *args, **kwargs)

    def callAxesFn(self, fname, *args, **kwargs):
        """Call any arbitrary function of current axes object."""
        if self.current_id < 0:
            self.addSubplot(1,1)
        fn = eval('self.axes[self.current_id].%s' % (fname))
        fn(*args, **kwargs)


class PlotView(PlotBase):
    """A default plotwidget implementation. This should be sufficient
    for most common usage.
    
    canvas: widget for plotting

    dataRoot: location of data tables

    """
    def __init__(self, *args, **kwargs):
        PlotBase.__init__(*args, **kwargs)
        self.canvas = CanvasWidget()
        self.dataRoot = '/data'
        
    def setDataRoot(self, path):
        self.dataRoot = path

    def addTimeSeries(self, table):        
        ts = np.linspace(0, moose.Clock('/clock').currentTime, len(table))
        self.canvas.plot(ts, table)

    def addRasterPlot(self, eventtable, yoffset=0, *args, **kwargs):
        """Add raster plot of events in eventtable.

        yoffset - offset along Y-axis.
        """
        y = np.ones(len(eventtable)) * yoffset
        self.canvas.plot(eventtable, y, '|')

    def getDataTablesPane(self):
        """This should create a tree widget with dataRoot as the root
        to allow visual selection of data tables for plotting."""
        raise NotImplementedError()



import sys
import os
import config
import unittest 

from PyQt4.QtTest import QTest 

class CanvasWidgetTests(unittest.TestCase):
    def setUp(self):
        self.app = QtGui.QApplication([])
        QtGui.qApp = self.app
        icon = QtGui.QIcon(os.path.join(config.KEY_ICON_DIR,'moose_icon.png'))
        self.app.setWindowIcon(icon)
        self.window = QtGui.QMainWindow()
        self.cwidget = CanvasWidget()
        self.window.setCentralWidget(self.cwidget)
        self.window.show()

    def testPlot(self):
        """Test plot function"""
        self.cwidget.addSubplot(1,1)
        self.cwidget.plot(np.arange(1000), mlab.normpdf(np.arange(1000), 500, 150))
        
    def testCallAxesFn(self):
        self.cwidget.addSubplot(1,1)
        self.cwidget.callAxesFn('scatter', np.random.randint(0, 100, 100), np.random.randint(0, 100,100))

    def tearDown(self):
        self.app.exec_()
    
if __name__ == '__main__':
    unittest.main()

# 
# mplot.py ends here
