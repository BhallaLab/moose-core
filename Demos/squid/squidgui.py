# squidgui.py --- 
# 
# Filename: squidgui.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Jul  9 18:23:55 2012 (+0530)
# Version: 
# Last-Updated: Mon Jul  9 22:56:06 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 236
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
    defaults = {}
    defaults.update(SquidAxon.defaults)
    defaults.update(ClampCircuit.defaults)
    defaults.update({'runtime': 50.0,
                  'simdt': 0.01,
                  'plotdt': 0.1,
                  'vclamp.holdingV': 0.0,
                  'vclamp.holdingT': 10.0,
                  'vclamp.prepulseV': 0.0,
                  'vclamp.prepulseT': 0.0,
                  'vclamp.clampV': 50.0,
                  'vclamp.clampT': 20.0,
                  'iclamp.baseI': 0.0,
                  'iclamp.firstI': 0.1,
                  'iclamp.firstT': 40.0,
                  'iclamp.firstD': 5.0,
                  'iclamp.secondI': 0.0,
                  'iclamp.secondT': 1e9,
                  'iclamp.secondD': 1e9
                  })
    def __init__(self, *args):
        QtGui.QMainWindow.__init__(self, *args)
        self.squid_setup = SquidSetup()
        self._plot_dict = defaultdict(list)
        self.setWindowTitle('Squid Axon simulation')        
        # self._statePlotWidget.setVisible(True)
        self._createRunControl()
        self._runControlBox.setWindowFlags(QtCore.Qt.Window)
        self._runControlBox.setWindowTitle('Simulation control')
        self._runControlBox.setVisible(True)
        self.setCentralWidget(self._runControlBox)
        self._createChannelControl()
        self._channelCtrlBox.setWindowFlags(QtCore.Qt.Window)
        self._channelCtrlBox.setWindowTitle('Channel properties')
        self._channelCtrlBox.setVisible(True)
        self._createElectronicsControl()
        self._electronicsTab.setWindowFlags(QtCore.Qt.Window)
        self._electronicsTab.setWindowTitle('Electronics')
        self._electronicsTab.setVisible(True)
        self._createPlotWidget()             
        self._plotWidget.setWindowFlags(QtCore.Qt.Window)
        self._plotWidget.setWindowTitle('Plots')
        self._plotWidget.setVisible(True)
        self._createStatePlotWidget()
        self._statePlotWidget.setWindowFlags(QtCore.Qt.Window)
        self._statePlotWidget.setWindowTitle('State plot')
        self._initActions()
        self._createToolBar()
        
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
        # self._statePlotAxes.legend()
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
        self._holdingVEdit = QtGui.QLineEdit(str(self.defaults['vclamp.holdingV']), vClampPanel)
        self._holdingTimeLabel = QtGui.QLabel("Holding Time (ms)", vClampPanel)
        self._holdingTimeEdit = QtGui.QLineEdit(str(self.defaults['vclamp.holdingT']), vClampPanel)
        self._prePulseVLabel = QtGui.QLabel("Pre-pulse Voltage (mV)", vClampPanel)
        self._prePulseVEdit = QtGui.QLineEdit(str(self.defaults['vclamp.prepulseV']), vClampPanel)
        self._prePulseTimeLabel = QtGui.QLabel("Pre-pulse Time (ms)", vClampPanel)
        self._prePulseTimeEdit = QtGui.QLineEdit(str(self.defaults['vclamp.prepulseT']), vClampPanel)
        self._clampVLabel = QtGui.QLabel("Clamp Voltage (mV)", vClampPanel)
        self._clampVEdit = QtGui.QLineEdit(str(self.defaults['vclamp.clampV']), vClampPanel)
        self._clampTimeLabel = QtGui.QLabel("Clamp Time (ms)", vClampPanel)
        self._clampTimeEdit = QtGui.QLineEdit(str(self.defaults['vclamp.clampT']), vClampPanel)
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
        self._baseCurrentEdit = QtGui.QLineEdit(str(self.defaults['iclamp.baseI']),iClampPanel)
        self._firstPulseLabel = QtGui.QLabel("First Pulse Current (uA)", iClampPanel)
        self._firstPulseEdit = QtGui.QLineEdit(str(self.defaults['iclamp.firstI']), iClampPanel)
        self._firstDelayLabel = QtGui.QLabel("First Onset Delay (ms)", iClampPanel)
        self._firstDelayEdit = QtGui.QLineEdit(str(self.defaults['iclamp.firstD']),iClampPanel)
        self._firstPulseWidthLabel = QtGui.QLabel("First Pulse Width (ms)", iClampPanel)
        self._firstPulseWidthEdit = QtGui.QLineEdit(str(self.defaults['iclamp.firstT']), iClampPanel)
        self._secondPulseLabel = QtGui.QLabel("Second Pulse Current (uA)", iClampPanel)
        self._secondPulseEdit = QtGui.QLineEdit(str(self.defaults['iclamp.secondI']), iClampPanel)
        self._secondDelayLabel = QtGui.QLabel("Second Onset Delay (ms)", iClampPanel)
        self._secondDelayEdit = QtGui.QLineEdit(str(self.defaults['iclamp.secondD']),iClampPanel)
        self._secondPulseWidthLabel = QtGui.QLabel("Second Pulse Width (ms)", iClampPanel)
        self._secondPulseWidthEdit = QtGui.QLineEdit(str(self.defaults['iclamp.secondT']), iClampPanel)
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

    def _overlayPlots(self, overlay):        
        if not overlay:
            for axis in (self._plotFigure.axes + self._statePlotFigure.axes):
                title = axis.get_title()
                axis.clear()
                axis.set_title(title)
            suffix = ''
        else:
            suffix = '_%d' % (len(self._plot_dict['vm']))
        self._vm_axes.set_xlim(0.0, self._runtime)
        self._g_axes.set_xlim(0.0, self._runtime)
        self._im_axes.set_xlim(0.0, self._runtime)
        self._i_axes.set_xlim(0.0, self._runtime)
        self._vm_plot, = self._vm_axes.plot([], [], label='Vm%s'%(suffix))
        self._plot_dict['vm'].append(self._vm_plot)
        self._command_plot, = self._vm_axes.plot([], [], label='command%s'%(suffix))
        self._plot_dict['command'].append(self._command_plot)
        # Channel conductances go to the same subplot
        self._gna_plot, = self._g_axes.plot([], [], label='Na%s'%(suffix))
        self._plot_dict['gna'].append(self._gna_plot)
        self._gk_plot, = self._g_axes.plot([], [], label='K%s'%(suffix))
        self._plot_dict['gk'].append(self._gk_plot)
        # Injection current for Vclamp/Iclamp go to the same subplot
        self._iclamp_plot, = self._im_axes.plot([], [], label='Iclamp%s'%(suffix))
        self._vclamp_plot, = self._im_axes.plot([], [], label='Vclamp%s'%(suffix))
        self._plot_dict['iclamp'].append(self._iclamp_plot)
        self._plot_dict['vclamp'].append(self._vclamp_plot)
        # Channel currents go to the same subplot
        self._ina_plot, = self._i_axes.plot([], [], label='Na%s'%(suffix))
        self._plot_dict['ina'].append(self._ina_plot)
        self._ik_plot, = self._i_axes.plot([], [], label='K%s'%(suffix))
        self._plot_dict['ik'].append(self._ik_plot)
        # self._i_axes.legend()
        # State plots
        self._state_plot, = self._statePlotAxes.plot([], [], label='state%s'%(suffix))
        self._plot_dict['state'].append(self._state_plot)
        self._m_plot, = self._activationParamAxes.plot([],[], label='m%s'%(suffix))
        self._h_plot, = self._activationParamAxes.plot([], [], label='h%s'%(suffix))
        self._n_plot, = self._activationParamAxes.plot([], [], label='n%s'%(suffix))
        self._plot_dict['m'].append(self._m_plot)
        self._plot_dict['h'].append(self._h_plot)
        self._plot_dict['n'].append(self._n_plot)
        if self._showLegendAction.isChecked():
            for axis in (self._plotFigure.axes + self._statePlotFigure.axes):            
                axis.legend()

    def _updateAllPlots(self):
        self._updatePlots()
        self._updateStatePlot()

    def _updatePlots(self):
        if len(self.squid_setup.vm_table.vec) <= 0:
            return        
        vm = numpy.asarray(self.squid_setup.vm_table.vec)
        cmd = numpy.asarray(self.squid_setup.cmd_table.vec)
        ik = numpy.asarray(self.squid_setup.ik_table.vec)
        ina = numpy.asarray(self.squid_setup.ina_table.vec)
        iclamp = numpy.asarray(self.squid_setup.iclamp_table.vec)
        vclamp = numpy.asarray(self.squid_setup.vclamp_table.vec)
        gk = numpy.asarray(self.squid_setup.gk_table.vec)
        gna = numpy.asarray(self.squid_setup.gna_table.vec)
        time_series = numpy.linspace(0, self._plotdt * len(vm), len(vm))        
        self._vm_plot.set_data(time_series, vm)
        time_series = numpy.linspace(0, self._plotdt * len(cmd), len(cmd))        
        self._command_plot.set_data(time_series, cmd)
        time_series = numpy.linspace(0, self._plotdt * len(ik), len(ik))
        self._ik_plot.set_data(time_series, ik)
        time_series = numpy.linspace(0, self._plotdt * len(ina), len(ina))
        self._ina_plot.set_data(time_series, ina)
        time_series = numpy.linspace(0, self._plotdt * len(iclamp), len(iclamp))
        self._iclamp_plot.set_data(time_series, iclamp)
        time_series = numpy.linspace(0, self._plotdt * len(vclamp), len(vclamp))
        self._vclamp_plot.set_data(time_series, vclamp)
        time_series = numpy.linspace(0, self._plotdt * len(gk), len(gk))
        self._gk_plot.set_data(time_series, gk)
        time_series = numpy.linspace(0, self._plotdt * len(gna), len(gna))
        self._gna_plot.set_data(time_series, gna)
        if self._autoscaleAction.isChecked():
            for axis in self._plotFigure.axes:
                axis.relim()
                axis.autoscale(True, axis='y')
        self._plotCanvas.draw()

    def _updateStatePlot(self):
        if len(self.squid_setup.vm_table.vec) <= 0:
            return
        sx = str(self._stateplot_xvar_combo.currentText())
        sy = str(self._stateplot_yvar_combo.currentText())
        xdata = self.__get_stateplot_data(sx)
        ydata = self.__get_stateplot_data(sy)
        minlen = min(len(xdata), len(ydata))
        self._state_plot.set_data(xdata[:minlen], ydata[:minlen])
        self._statePlotAxes.set_xlabel(sx)
        self._statePlotAxes.set_ylabel(sy)
        if sx == 'V':
            self._statePlotAxes.set_xlim(-20, 120)
        else:
            self._statePlotAxes.set_xlim(0, 1)
        if sy == 'V':
            self._statePlotAxes.set_ylim(-20, 120)
        else:
            self._statePlotAxes.set_ylim(0, 1)
        self._activationParamAxes.set_xlim(0, self._runtime)
        m = self.__get_stateplot_data('m')
        n = self.__get_stateplot_data('n')
        h = self.__get_stateplot_data('h')
        time_series = numpy.linspace(0, self._plotdt*len(m), len(m))
        self._m_plot.set_data(time_series, m)
        time_series = numpy.linspace(0, self._plotdt*len(h), len(h))
        self._h_plot.set_data(time_series, h)
        time_series = numpy.linspace(0, self._plotdt*len(n), len(n))
        self._n_plot.set_data(time_series, n)
        if self._autoscaleAction.isChecked():
            for axis in self._statePlotFigure.axes:
                axis.relim()
                axis.autoscale(True, axis='both')
        self._statePlotCanvas.draw()

    def _runSlot(self):
        if moose.isRunning():
            print 'Stopping simulation in progress ...'
            moose.stop()
        self._runtime = float(str(self._runTimeEdit.text()))
        self._overlayPlots(self._plotOverlayButton.isChecked())
        self._simdt = float(str(self._simTimeStepEdit.text()))
        self._plotdt = float(str(self._plotTimeStepEdit.text()))
        clampMode = None
        singlePulse = True
        if self._electronicsTab.currentWidget() == self._vClampCtrlBox:
            clampMode = 'vclamp'
            baseLevel = float(str(self._holdingVEdit.text()))
            firstDelay = float(str(self._holdingTimeEdit.text()))
            firstWidth = float(str(self._prePulseTimeEdit.text()))
            firstLevel = float(str(self._prePulseVEdit.text()))
            secondDelay = firstWidth
            secondWidth = float(str(self._clampTimeEdit.text()))
            secondLevel = float(str(self._clampVEdit.text()))
            self._im_axes.set_ylim(-10.0, 10.0)
        else:
            clampMode = 'iclamp'
            baseLevel = float(str(self._baseCurrentEdit.text()))
            firstDelay = float(str(self._firstDelayEdit.text()))
            firstWidth = float(str(self._firstPulseWidthEdit.text()))
            firstLevel = float(str(self._firstPulseEdit.text()))
            secondDelay = float(str(self._secondDelayEdit.text()))
            secondLevel = float(str(self._secondPulseEdit.text()))
            secondWidth = float(str(self._secondPulseWidthEdit.text()))
            singlePulse = (self._pulseMode.currentIndex() == 0)
            self._im_axes.set_ylim(-0.4, 0.4)
        self.squid_setup.clamp_ckt.configure_pulses(baseLevel=baseLevel,
                                                    firstDelay=firstDelay,
                                                    firstWidth=firstWidth,
                                                    firstLevel=firstLevel,
                                                    secondDelay=secondDelay,
                                                    secondWidth=secondWidth,
                                                    secondLevel=secondLevel,
                                                    singlePulse=singlePulse)
        if self._kConductanceToggle.isChecked():
            self.squid_setup.squid_axon.specific_gK = 0.0
        else:
            self.squid_setup.squid_axon.specific_gK = SquidAxon.defaults['specific_gK']
        if self._naConductanceToggle.isChecked():
            self.squid_setup.squid_axon.specific_gNa = 0.0
        else:
            self.squid_setup.squid_axon.specific_gNa = SquidAxon.defaults['specific_gNa']
        self.squid_setup.squid_axon.celsius = float(str(self._temperatureEdit.text()))
        self.squid_setup.squid_axon.K_out = float(str(self._kOutEdit.text()))
        self.squid_setup.squid_axon.Na_out = float(str(self._naOutEdit.text()))
        self.squid_setup.squid_axon.updateEk()
        self.squid_setup.schedule(self._simdt, self._plotdt, clampMode)
        # The following line is for use with Qthread
        self.squid_setup.run(self._runtime)
        self._updateAllPlots()

    def _initActions(self):
        self._runAction = QtGui.QAction(self.tr('Run'), self)
        self._runAction.setShortcut(self.tr('F5'))
        self._runAction.setToolTip('Run simulation (F5)')
        self.connect(self._runAction, QtCore.SIGNAL('triggered()'), self._runSlot)
        self._resetToDefaultsAction = QtGui.QAction(self.tr('Default settings'), self)
        self._resetToDefaultsAction.setToolTip('Reset all settings to their default values')
        self.connect(self._resetToDefaultsAction, QtCore.SIGNAL('triggered()'), self._useDefaults)
        self._showLegendAction = QtGui.QAction(self.tr('Display legend'), self)
        self._showLegendAction.setCheckable(True)
        self.connect(self._showLegendAction, QtCore.SIGNAL('toggled(bool)'), self._showLegend)
        self._showStatePlotAction = QtGui.QAction(self.tr('State plot'), self)
        self._showStatePlotAction.setCheckable(True)
        self._showStatePlotAction.setChecked(False)
        self.connect(self._showStatePlotAction, QtCore.SIGNAL('toggled(bool)'), self._statePlotWidget.setVisible)
        self._autoscaleAction  = QtGui.QAction(self.tr('Auto-scale plots'), self)
        self._autoscaleAction.setCheckable(True)
        self._autoscaleAction.setChecked(False)
        self.connect(self._autoscaleAction, QtCore.SIGNAL('toggled(bool)'), self._autoscale)
        self._useDefaultsAction = QtGui.QAction(self.tr('Use default values')
        self._quitAction = QtGui.QAction(self.tr('&Quit'), self)
        self._quitAction.setShortcut(self.tr('Ctrl+Q'))
        self.connect(self._quitAction, QtCore.SIGNAL('triggered()'), QtGui.qApp.closeAllWindows)

    def _createToolBar(self):
        self._simToolBar = self.addToolBar(self.tr('Simulation control'))
        self._simToolBar.addAction(self._quitAction)
        self._simToolBar.addAction(self._runAction)
        self._simToolBar.addAction(self._showLegendAction)
        self._simToolBar.addAction(self._autoscaleAction)
        self._simToolBar.addAction(self._showStatePlotAction)

    def _showLegend(self, on):
        if on:
            for axis in (self._plotFigure.axes + self._statePlotFigure.axes):            
                axis.legend().set_visible(True)
        else:
            for axis in (self._plotFigure.axes + self._statePlotFigure.axes):            
                axis.legend().set_visible(False)
        self._plotCanvas.draw()
        self._statePlotCanvas.draw()

    def _autoscale(self, on):
        if on:
            for axis in (self._plotFigure.axes + self._statePlotFigure.axes):            
                axis.relim()
                axis.autoscale(True, axis='y')
        else:
            for axis in self._plotFigure.axes:
                axis.autoscale(False)            
            self._vm_axes.set_ylim(-20.0, 120.0)
            self._g_axes.set_ylim(0.0, 0.5)
            self._im_axes.set_ylim(-0.5, 0.5)
            self._i_axes.set_ylim(-10, 10)
        self._plotCanvas.draw()
        self._statePlotCanvas.draw()
        

    def _useDefaults(self):
        self.squid_setup.use_defaults()

    def closeEvent(self, event):
        QtGui.qApp.closeAllWindows()
        
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    app.connect(app, QtCore.SIGNAL('lastWindowClosed()'), app, QtCore.SLOT('quit()'))
    QtGui.qApp = app
    squid_gui = SquidGui()
    squid_gui.show()
    print squid_gui.size()
    sys.exit(app.exec_())

# 
# squidgui.py ends here
