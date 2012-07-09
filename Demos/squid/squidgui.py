# squidgui.py --- 
# 
# Filename: squidgui.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Jul  9 18:23:55 2012 (+0530)
# Version: 
# Last-Updated: Mon Jul  9 19:03:40 2012 (+0530)
#           By: subha
#     Update #: 111
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

import sys
sys.path.append('../../python')
import os
os.environ['NUMPTHREADS'] = '1'

from collections import defaultdict
import time

from PyQt4 import QtGui
from PyQt4 import QtCore
import numpy
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QTAgg as NavigationToolbar

import moose

from squid import *
from squid_setup import SquidSetup
from electronics import ClampCircuit


class SquidGui(QtGui.QMainWindow):
    def __init__(self, *args):
        QtGui.QMainWindow.__init__(self, *args)
        self.squid_setup = SquidSetup()
        self.setWindowTitle('Squid Axon simulation')
        self.connect(self, QtCore.SIGNAL("destroyed()"), QtGui.qApp, QtCore.SLOT("closeAllWindows()"))
        self._createPlotWidget()             
        self._plotWidget.setWindowFlags(QtCore.Qt.Window)
        self.setCentralWidget(self._plotWidget)
        self._createStatePlotWidget()
        self._statePlotWidget.setWindowFlags(QtCore.Qt.Window)        
        # self._statePlotWidget.setVisible(True)
        self._createRunControl()
        self._runControlBox.setWindowFlags(QtCore.Qt.Window)
        self._runControlBox.setVisible(True)
        self._createChannelControl()
        self._channelCtrlBox.setWindowFlags(QtCore.Qt.Window)
        self._channelCtrlBox.setVisible(True)
        self._createElectronicsControl()
        self._electronicsTab.setWindowFlags(QtCore.Qt.Window)
        self._electronicsTab.setVisible(True)
        
    def _createPlotWidget(self):
        self._plotWidget = QtGui.QWidget()
        self._plotFigure = Figure()
        self._plotCanvas = FigureCanvas(self._plotFigure)
        self._plotCanvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self._plotCanvas.updateGeometry()
        self._plotCanvas.setParent(self._plotWidget)
        self._plotFigure.set_canvas(self._plotCanvas)
        # Vm and command voltage go in the same subplot
        self._vm_axes = self._plotFigure.add_subplot(2,2,1, title='Membrane potential')
        self._vm_axes.set_ylim(-20.0, 120.0)
        # self._vm_plot, = self._vm_axes.plot([], [], label='Vm')
        # self._command_plot, = self._vm_axes.plot([], [], label='command')
        # self._vm_axes.legend()
        # Channel conductances go to the same subplot
        self._g_axes = self._plotFigure.add_subplot(2,2,2, title='Channel conductance')
        self._g_axes.set_ylim(0.0, 0.5)
        # self._gna_plot, = self._g_axes.plot([], [], label='Na')
        # self._gk_plot, = self._g_axes.plot([], [], label='K')
        # self._g_axes.legend()
        # Injection current for Vclamp/Iclamp go to the same subplot
        self._im_axes = self._plotFigure.add_subplot(2,2,3, title='Injection current')
        self._im_axes.set_ylim(-0.5, 0.5)
        # self._iclamp_plot, = self._im_axes.plot([], [], label='Iclamp')
        # self._vclamp_plot, = self._im_axes.plot([], [], label='Vclamp')
        # self._im_axes.legend()
        # Channel currents go to the same subplot
        self._i_axes = self._plotFigure.add_subplot(2,2,4, title='Channel current')
        self._i_axes.set_ylim(-10, 10)
        # self._ina_plot, = self._i_axes.plot([], [], label='Na')
        # self._ik_plot, = self._i_axes.plot([], [], label='K')
        # self._i_axes.legend()
        for axis in self._plotFigure.axes:
            axis.autoscale(False)
        self._plotNavigator = NavigationToolbar(self._plotCanvas, self._plotWidget)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self._plotCanvas)
        layout.addWidget(self._plotNavigator)
        self._plotWidget.setLayout(layout)

    def _createStatePlotWidget(self):
        self._statePlotWidget = QtGui.QWidget()
        self._statePlotFigure = Figure()
        self._statePlotCanvas = FigureCanvas(self._statePlotFigure)
        self._statePlotCanvas.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self._statePlotCanvas.updateGeometry()
        self._statePlotCanvas.setParent(self._statePlotWidget)
        self._statePlotFigure.set_canvas(self._statePlotCanvas)
        self._statePlotFigure.subplots_adjust(hspace=0.5)
        self._statePlotAxes = self._statePlotFigure.add_subplot(2,1,1, title='State plot')
        self._state_plot, = self._statePlotAxes.plot([], [], label='state')
        self._statePlotAxes.legend()
        self._activationParamAxes = self._statePlotFigure.add_subplot(2,1,2, title='H-H activation parameters vs time')
        self._activationParamAxes.set_xlabel('Time (ms)')
        for axis in self._plotFigure.axes:
            axis.autoscale(False)
        # self._m_plot, = self._activationParamAxes.plot([],[], label='m')
        # self._h_plot, = self._activationParamAxes.plot([], [], label='h')
        # self._n_plot, = self._activationParamAxes.plot([], [], label='n')
        # self._activationParamAxes.legend()
        self._stateplot_xvar_label = QtGui.QLabel('Variable on X-axis')
        self._stateplot_xvar_combo = QtGui.QComboBox()
        self._stateplot_xvar_combo.addItems(['V', 'm', 'n', 'h'])
        self._stateplot_xvar_combo.setCurrentIndex(0)
        self._stateplot_xvar_combo.setEditable(False)
        self.connect(self._stateplot_xvar_combo,
                     QtCore.SIGNAL('currentIndexChanged(const QString&)'),
                     self._statePlotXSlot)                     
        self._stateplot_yvar_label = QtGui.QLabel('Variable on Y-axis')
        self._stateplot_yvar_combo = QtGui.QComboBox()
        self._stateplot_yvar_combo.addItems(['V', 'm', 'n', 'h'])
        self._stateplot_yvar_combo.setCurrentIndex(2)
        self._stateplot_yvar_combo.setEditable(False)
        self.connect(self._stateplot_yvar_combo,
                     QtCore.SIGNAL('currentIndexChanged(const QString&)'),
                     self._statePlotYSlot)
        self._statePlotNavigator = NavigationToolbar(self._statePlotCanvas, self._statePlotWidget)
        frame = QtGui.QFrame()
        layout = QtGui.QHBoxLayout()
        layout.addWidget(self._stateplot_xvar_label)
        layout.addWidget(self._stateplot_xvar_combo)
        layout.addWidget(self._stateplot_yvar_label)
        layout.addWidget(self._stateplot_yvar_combo)
        frame.setLayout(layout)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(frame)
        layout.addWidget(self._statePlotCanvas)
        layout.addWidget(self._statePlotNavigator)
        self._statePlotWidget.setLayout(layout)
        return self._statePlotWidget

    def _createRunControl(self):
        self._runControlBox = QtGui.QGroupBox('Simulation control')
        self._runTimeLabel = QtGui.QLabel("Run time (ms)", self._runControlBox)
        self._simTimeStepLabel = QtGui.QLabel("Simulation time step (ms)", self._runControlBox)
        self._plotTimeStepLabel = QtGui.QLabel("Plotting interval (ms)", self._runControlBox)
        self._runTimeEdit = QtGui.QLineEdit("50.0", self._runControlBox)
        self._simTimeStepEdit = QtGui.QLineEdit("0.01", self._runControlBox)
        self._plotTimeStepEdit = QtGui.QLineEdit("0.1", self._runControlBox)
        self._plotOverlayButton = QtGui.QCheckBox('Overlay plots', self._runControlBox)
        layout = QtGui.QGridLayout()
        layout.addWidget(self._runTimeLabel, 0, 0)
        layout.addWidget(self._runTimeEdit, 0, 1)
        layout.addWidget(self._simTimeStepLabel, 1, 0)
        layout.addWidget(self._simTimeStepEdit, 1, 1)
        layout.addWidget(self._plotTimeStepLabel, 2, 0)
        layout.addWidget(self._plotTimeStepEdit, 2, 1)
        layout.addWidget(self._plotOverlayButton, 3, 1)
        self._runControlBox.setLayout(layout)
        return self._runControlBox

    def _createChannelControl(self):
        self._channelCtrlBox = QtGui.QGroupBox('Channel properties', self)
        self._naConductanceToggle = QtGui.QCheckBox('Block Na+ channel', self._channelCtrlBox)
        self._kConductanceToggle = QtGui.QCheckBox('Block K+ channel', self._channelCtrlBox)
        self._kOutLabel = QtGui.QLabel('[K+]out (mM)', self._channelCtrlBox)
        self._kOutEdit = QtGui.QLineEdit(str(self.squid_setup.squid_axon.K_out), 
                                         self._channelCtrlBox)
        self._naOutLabel = QtGui.QLabel('[Na+]out (mM)', self._channelCtrlBox)
        self._naOutEdit = QtGui.QLineEdit(str(self.squid_setup.squid_axon.Na_out), 
                                         self._channelCtrlBox)
        self._kInLabel = QtGui.QLabel('[K+]in (mM)', self._channelCtrlBox)
        self._kInEdit = QtGui.QLabel(str(self.squid_setup.squid_axon.K_in), 
                                         self._channelCtrlBox)
        self._naInLabel = QtGui.QLabel('[Na+]in (mM)', self._channelCtrlBox)
        self._naInEdit = QtGui.QLabel(str(self.squid_setup.squid_axon.Na_in), 
                                         self._channelCtrlBox)
        self._temperatureLabel = QtGui.QLabel('Temperature (C)', self._channelCtrlBox)
        self._temperatureEdit = QtGui.QLineEdit(str(self.squid_setup.squid_axon.celsius),
                                                self._channelCtrlBox)
        for child in self._channelCtrlBox.children():
            if isinstance(child, QtGui.QLineEdit):
                child.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        layout = QtGui.QGridLayout(self._channelCtrlBox)
        layout.addWidget(self._naConductanceToggle, 0, 0)
        layout.addWidget(self._kConductanceToggle, 1, 0)
        layout.addWidget(self._naOutLabel, 2, 0)
        layout.addWidget(self._naOutEdit, 2, 1)
        layout.addWidget(self._naInLabel, 3, 0)
        layout.addWidget(self._naInEdit, 3, 1)
        layout.addWidget(self._kOutLabel, 4, 0)
        layout.addWidget(self._kOutEdit, 4, 1)
        layout.addWidget(self._kInLabel, 5, 0)
        layout.addWidget(self._kInEdit, 5, 1)
        layout.addWidget(self._temperatureLabel, 6, 0)
        layout.addWidget(self._temperatureEdit, 6, 1)
        self._channelCtrlBox.setLayout(layout)
        return self._channelCtrlBox
        

    def __get_stateplot_data(self, name):
        data = []
        if name == 'V':
            data = self.squid_setup.vm_table.vec
        elif name == 'm':
            data = self.squid_setup.m_table.vec
        elif name == 'h':
            data = self.squid_setup.h_table.vec
        elif name == 'n':
            data = self.squid_setup.n_table.vec
        else:
            raise ValueError('Unrecognized selection: %s' % (name))
        return numpy.asarray(data)
    
    def _statePlotYSlot(self, selectedItem):
        ydata = self.__get_stateplot_data(str(selectedItem))
        self._state_plot.set_ydata(ydata)
        self._statePlotAxes.set_ylabel(selectedItem)
        if str(selectedItem) == 'V':
            self._statePlotAxes.set_ylim(-20, 120)
        else:
            self._statePlotAxes.set_ylim(0, 1)
        self._statePlotCanvas.draw()
        
    def _statePlotXSlot(self, selectedItem):
        xdata = self.__get_stateplot_data(str(selectedItem))
        self._state_plot.set_xdata(xdata)
        self._statePlotAxes.set_xlabel(selectedItem)
        if str(selectedItem) == 'V':
            self._statePlotAxes.set_xlim(-20, 120)
        else:
            self._statePlotAxes.set_xlim(0, 1)
        self._statePlotCanvas.draw()

    def _createElectronicsControl(self):
        """Creates a tabbed widget of voltage clamp and current clamp controls"""
        self._electronicsTab = QtGui.QTabWidget(self)
        self._electronicsTab.addTab(self._getIClampCtrlBox(), 'Current clamp')
        self._electronicsTab.addTab(self._getVClampCtrlBox(), 'Voltage clamp')
        return self._electronicsTab

    def _getVClampCtrlBox(self):
        vClampPanel = QtGui.QGroupBox('Voltage clamp settings', self)
        self._vClampCtrlBox = vClampPanel
        self._holdingVLabel = QtGui.QLabel("Holding Voltage (mV)", vClampPanel)
        self._holdingVEdit = QtGui.QLineEdit("0.0", vClampPanel)
        self._holdingTimeLabel = QtGui.QLabel("Holding Time (ms)", vClampPanel)
        self._holdingTimeEdit = QtGui.QLineEdit("10.0", vClampPanel)
        self._prePulseVLabel = QtGui.QLabel("Pre-pulse Voltage (mV)", vClampPanel)
        self._prePulseVEdit = QtGui.QLineEdit("0.0", vClampPanel)
        self._prePulseTimeLabel = QtGui.QLabel("Pre-pulse Time (ms)", vClampPanel)
        self._prePulseTimeEdit = QtGui.QLineEdit("0.0", vClampPanel)
        self._clampVLabel = QtGui.QLabel("Clamp Voltage (mV)", vClampPanel)
        self._clampVEdit = QtGui.QLineEdit("50.0", vClampPanel)
        self._clampTimeLabel = QtGui.QLabel("Clamp Time (ms)", vClampPanel)
        self._clampTimeEdit = QtGui.QLineEdit("20.0", vClampPanel)
        for child in vClampPanel.children():
            if isinstance(child, QtGui.QLineEdit):
                child.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        layout = QtGui.QGridLayout(vClampPanel)
        layout.addWidget(self._holdingVLabel, 0, 0)
        layout.addWidget(self._holdingVEdit, 0, 1)
        layout.addWidget(self._holdingTimeLabel, 1, 0)
        layout.addWidget(self._holdingTimeEdit, 1, 1)
        layout.addWidget(self._prePulseVLabel, 2, 0)
        layout.addWidget(self._prePulseVEdit, 2, 1)
        layout.addWidget(self._prePulseTimeLabel,3,0)
        layout.addWidget(self._prePulseTimeEdit, 3, 1)
        layout.addWidget(self._clampVLabel, 4, 0)
        layout.addWidget(self._clampVEdit, 4, 1)
        layout.addWidget(self._clampTimeLabel, 5, 0)
        layout.addWidget(self._clampTimeEdit, 5, 1)
        vClampPanel.setLayout(layout)
        return self._vClampCtrlBox

    
    def _getIClampCtrlBox(self):
        iClampPanel = QtGui.QGroupBox('Current clamp settings', self)
        self._iClampCtrlBox = iClampPanel
        self._baseCurrentLabel = QtGui.QLabel("Base Current Level (uA)",iClampPanel)
        self._baseCurrentEdit = QtGui.QLineEdit("0.0",iClampPanel)
        self._firstPulseLabel = QtGui.QLabel("First Pulse Current (uA)", iClampPanel)
        self._firstPulseEdit = QtGui.QLineEdit("0.1", iClampPanel)
        self._firstDelayLabel = QtGui.QLabel("First Onset Delay (ms)", iClampPanel)
        self._firstDelayEdit = QtGui.QLineEdit("5.0",iClampPanel)
        self._firstPulseWidthLabel = QtGui.QLabel("First Pulse Width (ms)", iClampPanel)
        self._firstPulseWidthEdit = QtGui.QLineEdit("40.0", iClampPanel)
        self._secondPulseLabel = QtGui.QLabel("Second Pulse Current (uA)", iClampPanel)
        self._secondPulseEdit = QtGui.QLineEdit("0.0", iClampPanel)
        self._secondDelayLabel = QtGui.QLabel("Second Onset Delay (ms)", iClampPanel)
        self._secondDelayEdit = QtGui.QLineEdit("0.0",iClampPanel)
        self._secondPulseWidthLabel = QtGui.QLabel("Second Pulse Width (ms)", iClampPanel)
        self._secondPulseWidthEdit = QtGui.QLineEdit("0.0", iClampPanel)
        self._pulseMode = QtGui.QComboBox(iClampPanel)
        self._pulseMode.addItem("Single Pulse")
        self._pulseMode.addItem("Pulse Train")
        for child in iClampPanel.children():
            if isinstance(child, QtGui.QLineEdit):
                child.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        layout = QtGui.QGridLayout(iClampPanel)
        layout.addWidget(self._baseCurrentLabel, 0, 0)
        layout.addWidget(self._baseCurrentEdit, 0, 1)
        layout.addWidget(self._firstPulseLabel, 1, 0)
        layout.addWidget(self._firstPulseEdit, 1, 1)
        layout.addWidget(self._firstDelayLabel, 2, 0)
        layout.addWidget(self._firstDelayEdit, 2, 1)
        layout.addWidget(self._firstPulseWidthLabel, 3, 0)
        layout.addWidget(self._firstPulseWidthEdit, 3, 1)
        layout.addWidget(self._secondPulseLabel, 4, 0)
        layout.addWidget(self._secondPulseEdit, 4, 1)
        layout.addWidget(self._secondDelayLabel, 5, 0)
        layout.addWidget(self._secondDelayEdit, 5, 1)
        layout.addWidget(self._secondPulseWidthLabel, 6, 0)
        layout.addWidget(self._secondPulseWidthEdit, 6, 1)
        layout.addWidget(self._pulseMode, 7, 0, 1, 2)
        # layout.setSizeConstraint(QtGui.QLayout.SetFixedSize)
        iClampPanel.setLayout(layout)
        return self._iClampCtrlBox



if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    QtGui.qApp = app
    squid_gui = SquidGui()
    squid_gui.show()
    print squid_gui.size()
    sys.exit(app.exec_())

# 
# squidgui.py ends here
