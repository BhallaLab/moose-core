# pulsegengui.py --- 
# 
# Filename: pulsegengui.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Copyright (C) 2010 Subhasis Ray, all rights reserved.
# Created: Wed Dec 22 09:49:27 2010 (+0530)
# Version: 
# Last-Updated: Wed Dec 29 20:46:28 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 388
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# A simple gui to play with pulsegen.
# 
# 

# Change log:
# 
# 
# 

# Code:

import sys

from PyQt4.Qt import Qt
from PyQt4 import QtCore, QtGui
from PyQt4 import Qwt5 as Qwt

import numpy
import moose

RUNTIME = 200
TRIGGER_FIRST_LEVEL = 20
TRIGGER_FIRST_DELAY = 5
TRIGGER_FIRST_WIDTH = 1
PULSE_FIRST_LEVEL = 50
PULSE_FIRST_DELAY = 15
PULSE_FIRST_WIDTH = 3
class PulseGenWidget(QtGui.QWidget):
    """This widget contains a PulseGen object and a bunch of sliders
    and buttons to control its properties.

    One should be able to reuse this widget as part of a more
    complicated GUI.
    """
    def __init__(self, name='PulseGen',  parent=None):
        QtGui.QWidget.__init__(self, parent)
        print 'PulseGenDemo.__init__'
        self.controlPanel = QtGui.QFrame()
        self.controlPanel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)

        self.levelsPanel = QtGui.QFrame()
        self.levelsPanel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.baseLevelLabel = QtGui.QLabel('Base Level')
        self.firstLevelLabel = QtGui.QLabel('First Level')
        self.secondLevelLabel = QtGui.QLabel('Second Level')

        self.baseLevelSlider = QtGui.QSlider(Qt.Vertical)
        self.baseLevelSlider.setTickPosition(QtGui.QSlider.TicksLeft)
        self.baseLevelSlider.setMinimum(-100)
        self.baseLevelSlider.setMaximum(100)
        self.firstLevelSlider = QtGui.QSlider(Qt.Vertical)
        self.firstLevelSlider.setTickPosition(QtGui.QSlider.TicksLeft)
        self.firstLevelSlider.setMinimum(-100)
        self.firstLevelSlider.setMaximum(100)
        self.secondLevelSlider = QtGui.QSlider(Qt.Vertical)
        self.secondLevelSlider.setTickPosition(QtGui.QSlider.TicksLeft)
        self.secondLevelSlider.setMinimum(-100)
        self.secondLevelSlider.setMaximum(100)
        layout = QtGui.QGridLayout()
        layout.addWidget(self.baseLevelLabel, 0, 0)
        layout.addWidget(self.baseLevelSlider, 1, 0)
        layout.addWidget(self.firstLevelLabel, 0, 1)
        layout.addWidget(self.firstLevelSlider, 1, 1)
        layout.addWidget(self.secondLevelLabel, 0, 2)
        layout.addWidget(self.secondLevelSlider, 1, 2)
        self.levelsPanel.setLayout(layout)
        self.levelsPanel.setFrameStyle(QtGui.QFrame.StyledPanel)

        self.timesPanel = QtGui.QFrame()
        self.timesPanel.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.timesPanel.setFrameStyle(QtGui.QFrame.StyledPanel)
        self.firstDelayLabel = QtGui.QLabel('First Delay')
        self.firstWidthLabel = QtGui.QLabel('First Width')
        self.secondDelayLabel = QtGui.QLabel('Second Delay')
        self.secondWidthLabel = QtGui.QLabel('Second Width')        
        self.firstDelaySlider = QtGui.QSlider(Qt.Horizontal)
        self.firstDelaySlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.firstWidthSlider = QtGui.QSlider(Qt.Horizontal)
        self.firstWidthSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.secondDelaySlider = QtGui.QSlider(Qt.Horizontal)
        self.secondDelaySlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.secondWidthSlider = QtGui.QSlider(Qt.Horizontal)
        self.secondWidthSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.firstDelaySpinBox = QtGui.QSpinBox()
        self.firstWidthSpinBox = QtGui.QSpinBox()
        self.secondDelaySpinBox = QtGui.QSpinBox()
        self.secondWidthSpinBox = QtGui.QSpinBox()
        
        layout = QtGui.QGridLayout()

        layout.addWidget(self.firstDelayLabel, 0, 0)
        layout.addWidget(self.firstDelaySlider, 0, 1)
        layout.addWidget(self.firstDelaySpinBox, 0, 4)
        layout.addWidget(self.firstWidthLabel, 1, 0)
        layout.addWidget(self.firstWidthSpinBox, 1, 4)
        layout.addWidget(self.firstWidthSlider, 1, 1)
        layout.addWidget(self.firstWidthSpinBox, 1, 4)
        layout.addWidget(self.secondDelayLabel, 2, 0)
        layout.addWidget(self.secondDelaySlider, 2, 1)
        layout.addWidget(self.secondDelaySpinBox, 2, 4)        
        layout.addWidget(self.secondWidthLabel, 3, 0)
        layout.addWidget(self.secondWidthSlider, 3, 1)
        layout.addWidget(self.secondWidthSpinBox, 3, 4)
        self.timesPanel.setLayout(layout)

        self.modePanel = QtGui.QGroupBox('Trigger Mode')
        self.modePanel.setSizePolicy(QtGui.QSizePolicy.Maximum,QtGui.QSizePolicy.Maximum)
        self.modePanel.setFlat(False)
        self.freeRunButton = QtGui.QRadioButton('Free run')
        self.triggeredButton = QtGui.QRadioButton('Triggered')
        self.gatedButton = QtGui.QRadioButton('Gated')
        self.freeRunButton.setChecked(True)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.freeRunButton)
        layout.addWidget(self.triggeredButton)
        layout.addWidget(self.gatedButton)
        self.modePanel.setLayout(layout)

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.levelsPanel)
        layout.addWidget(self.timesPanel)
        layout.addWidget(self.modePanel)
        self.controlPanel.setLayout(layout)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.controlPanel)

        self.setLayout(layout)
        self.pulsegen = moose.PulseGen(str(name))

        self.connect(self.baseLevelSlider, QtCore.SIGNAL('valueChanged(int)'), self.changeBaseLevel)
        self.connect(self.firstLevelSlider, QtCore.SIGNAL('valueChanged(int)'), self.changeFirstLevel)
        self.connect(self.secondLevelSlider, QtCore.SIGNAL('valueChanged(int)'), self.changeSecondLevel)
        self.connect(self.firstDelaySlider, QtCore.SIGNAL('valueChanged(int)'), self.changeFirstDelay)
        self.connect(self.firstWidthSlider, QtCore.SIGNAL('valueChanged(int)'), self.changeFirstWidth)
        self.connect(self.secondDelaySlider, QtCore.SIGNAL('valueChanged(int)'), self.changeSecondDelay)
        self.connect(self.secondWidthSlider, QtCore.SIGNAL('valueChanged(int)'), self.changeSecondWidth)
        self.connect(self.freeRunButton, QtCore.SIGNAL('toggled(bool)'), self.changeRunMode)
        self.connect(self.triggeredButton, QtCore.SIGNAL('toggled(bool)'), self.changeRunMode)
        self.connect(self.gatedButton, QtCore.SIGNAL('toggled(bool)'), self.changeRunMode)
        self.connect(self.firstDelaySlider, QtCore.SIGNAL('valueChanged(int)'), self.firstDelaySpinBox.setValue)
        self.connect(self.firstWidthSlider, QtCore.SIGNAL('valueChanged(int)'), self.firstWidthSpinBox.setValue)
        self.connect(self.secondDelaySlider, QtCore.SIGNAL('valueChanged(int)'), self.secondDelaySpinBox.setValue)
        self.connect(self.secondWidthSlider, QtCore.SIGNAL('valueChanged(int)'), self.secondWidthSpinBox.setValue)
        self.connect(self.firstDelaySpinBox, QtCore.SIGNAL('valueChanged(int)'), self.firstDelaySlider.setValue)
        self.connect(self.firstWidthSpinBox, QtCore.SIGNAL('valueChanged(int)'), self.firstWidthSlider.setValue)
        self.connect(self.secondDelaySpinBox, QtCore.SIGNAL('valueChanged(int)'), self.secondDelaySlider.setValue)
        self.connect(self.secondWidthSpinBox, QtCore.SIGNAL('valueChanged(int)'), self.secondWidthSlider.setValue)

        
    def changeBaseLevel(self, value):
        self.pulsegen.baseLevel = value * 1.0

    def changeFirstLevel(self, value):
        self.pulsegen.firstLevel = value * 1.0

    def changeSecondLevel(self, value):
        self.pulsegen.secondLevel = value * 1.0

    def changeFirstDelay(self, value):
        self.pulsegen.firstDelay = value * 1.0

    def changeSecondDelay(self, value):
        self.pulsegen.secondDelay = value * 1.0

    def changeFirstWidth(self, value):
        self.pulsegen.firstWidth = value * 1.0

    def changeSecondWidth(self, value):
        self.pulsegen.secondWidth = value * 1.0

    def changeRunMode(self, toggled):
        if self.freeRunButton.isChecked():
            self.pulsegen.trigMode = 0
        elif self.triggeredButton.isChecked():
            self.pulsegen.trigMode = 1
        elif self.gatedButton.isChecked():
            self.pulsegen.trigMode = 2
        else:
            raise Exception('This should never be reached')


class PulseGenDemo(QtGui.QWidget):
    """Demo class wrapping a trigger and a pulsegen with controls."""
    def __init__(self, *args):
        QtGui.QWidget.__init__(self, *args)
        self.controlPanel = QtGui.QFrame()
        
        self.runButton = QtGui.QPushButton('Run')
        self.runtimeLabel = QtGui.QLabel('Runtime')
        self.runtimeSlider = QtGui.QSlider()
        self.runtimeSlider.setMaximum(RUNTIME)
        self.runtimeSlider.setMinimum(0)
        self.runtimeSlider.setSliderPosition(RUNTIME)
        runControlWidget = QtGui.QFrame()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.runButton)
        layout.addWidget(self.runtimeLabel)
        layout.addWidget(self.runtimeSlider)
        runControlWidget.setLayout(layout)
        runControlWidget.setSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Minimum)
        pulsegen_gb = QtGui.QGroupBox('PulseGen')
        self.pulseGenWidget = PulseGenWidget('PulseGen', pulsegen_gb)
        self.pulseGenWidget.firstDelaySlider.setSliderPosition(PULSE_FIRST_DELAY)
        self.pulseGenWidget.firstWidthSlider.setSliderPosition(PULSE_FIRST_WIDTH)
        self.pulseGenWidget.firstLevelSlider.setSliderPosition(PULSE_FIRST_LEVEL)
        self.pulseGenWidget.triggeredButton.setChecked(True)
        pulsegen_gb.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        trigger_gb = QtGui.QGroupBox('Trigger')
        self.triggerWidget = PulseGenWidget('Trigger', parent=trigger_gb)
        self.triggerWidget.firstDelaySlider.setSliderPosition(TRIGGER_FIRST_DELAY)
        self.triggerWidget.firstWidthSlider.setSliderPosition(TRIGGER_FIRST_WIDTH)
        self.triggerWidget.firstLevelSlider.setSliderPosition(TRIGGER_FIRST_LEVEL)
        trigger_gb.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
        layout = QtGui.QVBoxLayout()
        layout.addWidget(trigger_gb)
        layout.addWidget(pulsegen_gb)
        
        pulseControlWiget = QtGui.QFrame()
        pulseControlWiget.setFrameStyle(QtGui.QFrame.StyledPanel)
        pulseControlWiget.setLayout(layout)
        pulseControlWiget.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        layout = QtGui.QHBoxLayout()
        layout.addWidget(pulseControlWiget)
        layout.addWidget(runControlWidget)
        self.controlPanel.setLayout(layout)
        
        self.plot = Qwt.QwtPlot()
        self.curve = Qwt.QwtPlotCurve('PulseGen')
        self.curve.setPen(Qt.red)
        self.curve.attach(self.plot)
        self.trigCurve = Qwt.QwtPlotCurve('Trigger')
        self.trigCurve.setPen(Qt.green)
        self.trigCurve.attach(self.plot)
        self.plot.insertLegend(Qwt.QwtLegend(), Qwt.QwtPlot.RightLegend)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.controlPanel)
        layout.addWidget(self.plot)
        self.setLayout(layout)
        if not moose.context.exists('/data'):
            moose.Neutral('/data')
        self.table = moose.Table('/data/pulsegen')
        self.table.stepMode = 3
        self.table.connect('inputRequest', self.pulseGenWidget.pulsegen, 'output')
        self.trigTable = moose.Table('/data/trigger')
        self.trigTable.stepMode = 3
        self.trigTable.connect('inputRequest', self.triggerWidget.pulsegen, 'output')
        self.triggerWidget.pulsegen.connect('outputSrc', self.pulseGenWidget.pulsegen, 'input')

        self.connect(self.runtimeSlider, QtCore.SIGNAL('valueChanged(int)'), self.setRuntime)
        self.connect(self.runButton, QtCore.SIGNAL('clicked()'), self.doRun)

        self.runtime = RUNTIME * 1.0

    def setRuntime(self, value):
        self.runtime = value * 1.0

    def doRun(self):
        moose.context.reset()
        moose.context.step(self.runtime)
        self.curve.setData(numpy.linspace(0, self.runtime, len(self.table)), numpy.array(self.table))
        self.trigCurve.setData(numpy.linspace(0, self.runtime, len(self.trigTable)), numpy.array(self.trigTable))
        self.plot.replot()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    # widget = PulseGenWidget()
    widget = PulseGenDemo()
    widget.resize(800, 600)
    # print '#', widget.pulseGenWidget
    # widget.table.connect('inputRequest', widget.pulseGenWidget.pulsegen, 'output')
    widget.show()
    sys.exit(app.exec_())
    print 'Finished'

# 
# pulsegengui.py ends here
