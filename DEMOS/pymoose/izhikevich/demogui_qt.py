# demo_gui.py --- 
# 
# Filename: demo_gui.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed Jun 16 05:41:58 2010 (+0530)
# Version: 
# Last-Updated: Wed Jun 16 09:53:00 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 192
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

from PyQt4 import QtGui, QtCore
import PyQt4.Qwt5 as Qwt
import numpy
import pylab
from Izhikevich import IzhikevichDemo


class IzhikevichGui(QtGui.QMainWindow):
    """This is a Qt version of the GUI"""
    def __init__(self, *args):
        QtGui.QMainWindow.__init__(self, *args)
        self.demo = IzhikevichDemo()
        self.signalMapper = QtCore.QSignalMapper(self)
        self.demoFrame = QtGui.QFrame(self)
        self.controlPanel = QtGui.QFrame(self.demoFrame)
        self.figureNo = {}
        self.buttons = {}
        for key, value in IzhikevichDemo.parameters.items():
            button = QtGui.QPushButton(key, self.controlPanel)
            self.figureNo[value[0]] = key
            self.buttons[key] = button
        keys = self.figureNo.keys()
        keys.sort()
        length = len(keys)
        rows = int(numpy.rint(numpy.sqrt(length)))
        cols = int(numpy.ceil(length * 1.0 / rows))
        layout = QtGui.QGridLayout()
        for ii in range(rows):
            for jj in range(cols):
                index = ii * cols + jj
                if  index < length:
                    key = self.figureNo[keys[index]]
                    button = self.buttons[key]
                    print 'Adding button ', button.text(), 'at (', ii, ',', jj, ')'
                    layout.addWidget(button, ii, jj)
                    self.connect(button, QtCore.SIGNAL('clicked()'), self.signalMapper, QtCore.SLOT('map()'))
                    self.signalMapper.setMapping(button, key)

        self.connect(self.signalMapper, QtCore.SIGNAL('mapped(const QString &)'), self._simulateAndPlot)         
        self.controlPanel.setLayout(layout)
        self.plotPanel = QtGui.QFrame(self.demoFrame)
        self.plot = Qwt.QwtPlot(self.plotPanel)
        self.plot.setAxisTitle(Qwt.QwtPlot.xBottom, 'time (ms)')
        self.plot.setAxisTitle(Qwt.QwtPlot.yLeft, 'Vm (mV)')
        self.plot.replot()
        layout = QtGui.QVBoxLayout(self.demoFrame)
        layout.addWidget(self.plot)
        self.plotPanel.setLayout(layout)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.plotPanel)
        layout.addWidget(self.controlPanel)
        self.demoFrame.setLayout(layout)
        self.setCentralWidget(self.demoFrame)

    def _simulateAndPlot(self, key):
        key = str(key)
        (time, Vm, Im) = self.demo.simulate(key)
        Vm = numpy.array(Vm) * 1e3
        Im = numpy.array(Im) * 1e9 - 120.0
        numpy.savetxt(key + '_Vm.plot', Vm)
        numpy.savetxt(key + '_Im.plot', Im)
        self.plot.clear()
        curve = Qwt.QwtPlotCurve(self.tr(key + '_Vm'))
        curve.setPen(QtCore.Qt.red)
        curve.setData(time, numpy.array(Vm))
        curve.attach(self.plot)
        curve = Qwt.QwtPlotCurve(self.tr(key + '_Im'))
        curve.setPen(QtCore.Qt.blue)
        curve.setData(time, Im)
        curve.attach(self.plot)
        self.plot.replot()

import sys
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    mainWin = IzhikevichGui()
    mainWin.show()
    sys.exit(app.exec_())

# 
# demo_gui.py ends here
