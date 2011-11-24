# glwizard.py --- 
# 
# Filename: glwizard.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Fri Jul  9 21:23:39 2010 (+0530)
# Version: 
# Last-Updated: Thu Jul 22 14:07:05 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 846
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# This class provides a simple wizard for 3-D visualization of MOOSE
# simulations using GLcell/GLview.
# 
# The first page is for selecting the glcell/glview mode and selecting
# the target object and the target parameter.
# The second page checks the client executable and starts it. 

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

import os
import time

from PyQt4.Qt import Qt
from PyQt4 import QtCore, QtGui

import moose
import config
from moosehandler import MooseHandler
from moosetree import MooseTreeWidget


class MooseGLWizard(QtGui.QWizard):
    """Wizard class for starting a glclient and creating glcell for
    moose model element."""
    currentPort = int(config.GL_PORT)
    pageIdMap = {'glclient': 0,
                 'glcell': 1,
                 'glview': 2}

    def __init__(self, parent=None, mooseHandler=None):
        QtGui.QWizard.__init__(self, parent)
        if not mooseHandler:
            mooseHandler = MooseHandler()
        self._mooseHandler = mooseHandler
        self._port = MooseGLWizard.currentPort
        self._targetObjectPath = '/'
        MooseGLWizard.currentPort += 1
        self.setPage(MooseGLWizard.pageIdMap['glclient'], GLClientWizardPage())
        self.setPage(MooseGLWizard.pageIdMap['glcell'], GLCellWizardPage())
        self.setPage(MooseGLWizard.pageIdMap['glview'], GLViewWizardPage())
        self.connect(self, QtCore.SIGNAL('accepted()'), self.createGLSlot)
        
    def createGLSlot(self):
        bgColor = self.field('bgColor').toString()
        print 'Background colour', bgColor
        color = QtGui.QColor()
        color.setNamedColor(bgColor)
        bgColor = '%03d%03d%03d' % (color.red(), color.green(), color.blue())
        sync = 'on' if self.field('syncButton').toBool() else 'off'
        glcellMode = self.field('glcellMode').toBool()
        port = str(self.field('port').toString())
        glclient = str(self.field('glclient').toString())
        colormap = str(self.field('colormap').toString())
        print 'Starting client with the following parameters: background: %s\nsync: %s\nglCellMode: %d\nport: %s\nglclient: %s\ncolormap: %s' % (bgColor, sync, glcellMode, port, glclient, colormap)
        try:
            client = self._mooseHandler.startGLClient(glclient, port, 'c' if glcellMode else 'v', colormap)
        except OSError, e:
            print e
            return

        if client is None:
            return

        self._port += 1
        self.setField(QtCore.QString('port'), QtCore.QVariant(self._port))
        time.sleep(3)
        if glcellMode:
            field = str(self.field('glcellfield').toString())
            (threshold, ok) = self.field('threshold').toDouble()
            if not ok:
                threshold = None
                
            (highValue, ok) = self.field('highValue').toDouble()
            if not ok:
                highValue = None
            (lowValue, ok) = self.field('lowValue').toDouble()
            if not ok:
                lowValue = None
            (vscale, ok) = self.field('vscale').toDouble()
            if not ok:
                vscale = None                
            target = self._targetObjectPath#str(self.field('targetObject'))
            print 'Target object', target
            self._mooseHandler.makeGLCell(target, port, field, threshold, lowValue, highValue, vscale, bgColor, sync)
        else:
            field = []
            valueMin = []
            valueMax = []
            for ii in range(5):
                field.append(str(self.field('fieldName%d' % (ii+1)).toString()))
                (value, ok) = self.field('valueMin%d' % (ii+1)).toDouble()
                if not ok:
                    value = 0.0
                valueMin.append(value)
                (value, ok) = self.field('valueMax%d' % (ii+1)).toDouble() 
                if not ok:
                    value = 1.0
                valueMax.append(value)
	    wildcard = str(self.field('wildcard').toString())
            (colorFieldIndex, ok) = self.field('colorField').toInt()
            if not ok:
                colorFieldIndex = 1
            (morphFieldIndex, ok) = self.field('morphField').toInt()
            if not ok:
                morphFieldIndex = 1
            grid = 'on' if self.field('grid').toBool() else 'off'
            self._mooseHandler.makeGLView(self._targetObjectPath, wildcard, port, field, valueMin, valueMax, colorFieldIndex, morphFieldIndex, grid, bgColor, sync)

class GLClientWizardPage(QtGui.QWizardPage):
    def __init__(self, *args):
        QtGui.QWizardPage.__init__(self, *args)
        self.setTitle(self.tr('Configure GL Client'))
        self.setSubTitle(self.tr('Setup the client for 3-D visualization'))        
        exeLabel = QtGui.QLabel(self.tr('&Executable of glclient:'))
        glClientExe = config.get_settings().value(config.KEY_GL_CLIENT_EXECUTABLE).toString()
        if glClientExe:
            glClientExe = unicode(glClientExe)
        else:
            glClientExe = config.GL_CLIENT_EXECUTABLE
        
        exeLineEdit = QtGui.QLineEdit(str(glClientExe))
        exeButton = QtGui.QPushButton(self.tr('Browse'))
        self.connect(exeButton, QtCore.SIGNAL('clicked()'), self._selectExeSlot)
        exeLabel.setBuddy(exeButton)
        colormap = config.get_settings().value(config.KEY_GL_COLORMAP).toString()
        if colormap:
            colormap = unicode(colormap)
        else:
            colormap = config.GL_DEFAULT_COLORMAP
        colormapLabel = QtGui.QLabel(self.tr('&Colormap File'))
        colormapLineEdit = QtGui.QLineEdit(colormap)
        colormapButton = QtGui.QPushButton(self.tr('Browse'))
        self.connect(colormapButton, QtCore.SIGNAL('clicked()'), self._selectColormapSlot)
        colormapLabel.setBuddy(colormapButton)

        frame = QtGui.QFrame()
        frame.setFrameStyle(QtGui.QFrame.Raised | QtGui.QFrame.StyledPanel)
        frame.setLayout(QtGui.QVBoxLayout())
        modeGroupBox = QtGui.QGroupBox(self.tr('Mode'), frame)
        frame.layout().addWidget(modeGroupBox)

        glCellRadioButton = QtGui.QRadioButton(self.tr('GLCell'))
        glViewRadioButton = QtGui.QRadioButton(self.tr('GLView'))
        layout = QtGui.QHBoxLayout()
        layout.addWidget(glCellRadioButton)
        layout.addWidget(glViewRadioButton)
        glCellRadioButton.setChecked(True)
        modeGroupBox.setLayout(layout)
        
        portLabel = QtGui.QLabel(self.tr('&Port'))
        port = self.field(QtCore.QString('port')).toString()
        if port:
            port = str(port)
        else:
            port = config.get_settings().value(config.KEY_GL_PORT).toString()
        portLineEdit = QtGui.QLineEdit(str(port))
        portLabel.setBuddy(portLineEdit)
        
        syncButton = QtGui.QCheckBox(self.tr('&Sync'))
        syncButton.setToolTip(self.tr('Slow down the simulation to keep pace with the 3-D rendering.'))
        syncButton.setChecked(False)

        bgColorLabel = QtGui.QLabel(self.tr('&Background Color'))
        color = config.get_settings().value(QtCore.QString(config.KEY_GL_BACKGROUND_COLOR)).toString()
        if color:
            color = str(color)
        else:
            color = QtGui.QColor(Qt.black).name()
        bgColorButton = QtGui.QPushButton(color)
        bgColorButton.setObjectName('bgColorButton')
        styleStr = QtCore.QString('QPushButton#bgColorButton {Background-color: %s}' % color)
        bgColorButton.setStyleSheet(styleStr)
        self.connect(bgColorButton, QtCore.SIGNAL('clicked()'), self._chooseBackgroundColorSlot)
        bgColorLabel.setBuddy(bgColorButton)

        layout = QtGui.QGridLayout(self)
        layout.addWidget(exeLabel, 0, 0)
        layout.addWidget(exeButton, 0, 1)
        layout.addWidget(exeLineEdit, 0, 2, 1, -1)

        layout.addWidget(colormapLabel, 1, 0)
        layout.addWidget(colormapButton, 1, 1)
        layout.addWidget(colormapLineEdit, 1, 2, 1, -1)
        
        layout.addWidget(frame, 2, 0, 1, 4)
        layout.addWidget(portLabel, 3, 0)
        layout.addWidget(portLineEdit, 3, 1)

        layout.addWidget(syncButton, 4, 0)
        layout.addWidget(bgColorLabel, 4, 1)
        layout.addWidget(bgColorButton, 4, 2)

        self.registerField('glclient', exeLineEdit)
        self.registerField('colormap', colormapLineEdit)
        self.registerField('port', portLineEdit)
        self.registerField('glcellMode', glCellRadioButton)
        self.registerField('bgColor', bgColorButton, 'text')
        self.registerField('syncButton', syncButton)
        self.setField(QtCore.QString('glclient'), QtCore.QVariant(exeLineEdit.text()))
        self.setField(QtCore.QString('colormap'), QtCore.QVariant(colormapLineEdit.text()))
        self.setField(QtCore.QString('port'), QtCore.QVariant(portLineEdit.text()))
        self.setField(QtCore.QString('glCellMode'), QtCore.QVariant(glCellRadioButton.isChecked()))
        self.setField(QtCore.QString('bgColor'), QtCore.QVariant(bgColorButton.text()))
        self.setField(QtCore.QString('syncButton'), QtCore.QVariant(syncButton.isChecked()))
        self._exeLineEdit = exeLineEdit
        self._colormapLineEdit = colormapLineEdit
        self._bgColorButton = bgColorButton

    def _selectColormapSlot(self):
        colormap = unicode(QtGui.QFileDialog.getOpenFileName(self, 'Select Colormap File'))
        self._colormapLineEdit.setText(colormap)
        config.get_settings().setValue(config.KEY_GL_COLORMAP, colormap)

    def _selectExeSlot(self):
        self._glClientExe = unicode(QtGui.QFileDialog.getOpenFileName(self, 'Select glclient executable'))
        self._exeLineEdit.setText(self._glClientExe)
        config.get_settings().setValue(config.KEY_GL_CLIENT_EXECUTABLE, self._glClientExe)

    def _chooseBackgroundColorSlot(self):
        currentColor = self._bgColorButton.palette().color(QtGui.QPalette.Background)
        color = QtGui.QColorDialog.getColor(currentColor, self)
        if color.isValid():
            style = QtCore.QString('QPushButton#bgColorButton {Background-color: %s}' % color.name())
            self._bgColorButton.setStyleSheet(style)
            self._bgColorButton.setText(color.name())
            self.setField(QtCore.QString('bgColor'), QtCore.QVariant(color.name()))
            config.get_settings().setValue(QtCore.QString(config.KEY_GL_BACKGROUND_COLOR), QtCore.QVariant(color.name()))
    
    def nextId(glClientPage):
	if glClientPage.field('glcellMode').toBool():
		return MooseGLWizard.pageIdMap['glcell']  
	else:
		return MooseGLWizard.pageIdMap['glview']


class GLCellWizardPage(QtGui.QWizardPage):
    def __init__(self, *args):
        QtGui.QWizardPage.__init__(self, *args)
        self.setTitle(self.tr('Select GLCell Target '))
        self.setSubTitle(self.tr('Expand the model tree and select an element to observe in 3-D'))
        self._targetObjectPath = '/'
        layout = QtGui.QGridLayout()
        tree = MooseTreeWidget()
        self.connect(tree, QtCore.SIGNAL('itemClicked(QTreeWidgetItem*, int)'), self._setTargetObject)

        fieldsLabel = QtGui.QLabel(self.tr('&Field to observe'))
        fieldsLineEdit = QtGui.QLineEdit('Vm')
        fieldsLabel.setBuddy(fieldsLineEdit)

        frame = QtGui.QFrame()
        frame.setLayout(QtGui.QHBoxLayout())
        frame.setFrameStyle(QtGui.QFrame.Raised | QtGui.QFrame.StyledPanel)
        frame.layout().addWidget(fieldsLabel)
        frame.layout().addWidget(fieldsLineEdit)

        thresholdLabel = QtGui.QLabel(self.tr('&Threshold (%)'))
        thresholdLabel.setToolTip(self.tr('The percentage change in the field that should be picked up by the 3D visualizer.'))
        thresholdLineEdit = QtGui.QLineEdit('1')
        thresholdLabel.setBuddy(thresholdLineEdit)

        highValueLabel = QtGui.QLabel(self.tr('&High Value'))
        highValueLabel.setToolTip(self.tr('The value of the field corresponding the to hottest colour'))
        highValueLineEdit = QtGui.QLineEdit('50e-3')
        highValueLabel.setBuddy(highValueLineEdit)

        lowValueLabel = QtGui.QLabel(self.tr('&Low Value'))
        lowValueLabel.setToolTip(self.tr('The value of the field corresponding the to coolest colour'))
        lowValueLineEdit = QtGui.QLineEdit('-120e-3')
        lowValueLabel.setBuddy(lowValueLineEdit)

        vscaleLabel = QtGui.QLabel(self.tr('&Vscale'))
        vscaleLineEdit = QtGui.QLineEdit(self.tr('1.0'))
        vscaleLineEdit.setToolTip(self.tr('Scale up the thickness of compartments for visualization.'))
        vscaleLabel.setBuddy(vscaleLineEdit)
        
        layout.addWidget(tree, 0, 0, 5, 5)
        layout.addWidget(frame, 4, 5, 1, 3)
        layout.addWidget(thresholdLabel, 5, 0)
        layout.addWidget(thresholdLineEdit, 5, 1, 1, 2)
        layout.addWidget(vscaleLabel, 5, 3)
        layout.addWidget(vscaleLineEdit, 5, 4, 1, 2)
        layout.addWidget(lowValueLabel, 6, 0)
        layout.addWidget(lowValueLineEdit, 6, 1, 1, 2)
        layout.addWidget(highValueLabel, 6, 3)
        layout.addWidget(highValueLineEdit, 6, 4, 1, 2)
        
        self.setLayout(layout)
        self.registerField('glcellfield', fieldsLineEdit)
        self.registerField('threshold', thresholdLineEdit)
        self.registerField('vscale', vscaleLineEdit)
        self.registerField('highValue', highValueLineEdit)
        self.registerField('lowValue', lowValueLineEdit)
        self.setField(QtCore.QString('glcellfield'), QtCore.QVariant(fieldsLineEdit.text()))
        self.setField(QtCore.QString('threshold'), QtCore.QVariant(thresholdLineEdit.text()))
        self.setField(QtCore.QString('vscale'), QtCore.QVariant(vscaleLineEdit.text()))
        self.setField(QtCore.QString('highValue'), QtCore.QVariant(highValueLineEdit.text()))
        self.setField(QtCore.QString('lowValue'), QtCore.QVariant(lowValueLineEdit.text()))
        
        self._fieldsLineEdit = fieldsLineEdit

    def nextId(self):
        return -1

    def _setTargetObject(self, item, column):
        self.setField(QtCore.QString('targetObject'), QtCore.QVariant(QtCore.QString(item.getMooseObject().path)))
        self.wizard()._targetObjectPath = item.getMooseObject().path

class GLViewWizardPage(QtGui.QWizardPage):
    def __init__(self, *args):
        QtGui.QWizardPage.__init__(self, *args)
        self.setTitle(self.tr('Select GLView Target'))
        self.setSubTitle(self.tr('Expand the model tree and select an element to observe in 3-D'))
        tree = MooseTreeWidget()
        self.connect(tree, QtCore.SIGNAL('itemClicked(QTreeWidgetItem*, int)'), self._setTargetObject)

        wildCardLabel = QtGui.QLabel(self.tr('&Wildcard for elements to observe'))
        wildCardLineEdit = QtGui.QLineEdit('##[CLASS=Compartment]')
        wildCardLabel.setBuddy(wildCardLineEdit)
        
        frame = QtGui.QFrame()
        frame.setFrameStyle(QtGui.QFrame.Raised | QtGui.QFrame.StyledPanel)
        

        layout = QtGui.QGridLayout()
        currentRow = 0        
        layout.addWidget(wildCardLabel, currentRow, 0)
        layout.addWidget(wildCardLineEdit, currentRow, 1, 1, 3)
        currentRow += 1
        layout.addWidget(QtGui.QLabel('Enter Fields to Observe'), currentRow, 0, 1, 2)
        currentRow += 1
        layout.addWidget(QtGui.QLabel('Field #'), currentRow, 0)
        layout.addWidget(QtGui.QLabel('FieldName'), currentRow, 1)
        layout.addWidget(QtGui.QLabel('Minimum Value'), currentRow, 2)
        layout.addWidget(QtGui.QLabel('Maximum Value'), currentRow, 3)
        valueFieldMap = {}
        currentRow += 1
        for ii in range(5):
            valueLabel = QtGui.QLabel(self.tr('Field# %d' % (ii+1)))
            fieldNameEdit = QtGui.QLineEdit()
            valueMinEdit = QtGui.QLineEdit()
            valueMaxEdit = QtGui.QLineEdit()
            valueFieldMap[ii+1] = (valueLabel, fieldNameEdit, valueMinEdit, valueMaxEdit)
            layout.addWidget(valueLabel, currentRow, 0)
            layout.addWidget(fieldNameEdit, currentRow, 1)
            layout.addWidget(valueMinEdit, currentRow, 2)
            layout.addWidget(valueMaxEdit, currentRow, 3)                    
            currentRow += 1
        sep = QtGui.QFrame()
        sep.setFrameStyle(QtGui.QFrame.Sunken | QtGui.QFrame.VLine)
        colorFieldLabel = QtGui.QLabel(self.tr('Map Color to Field #'))
        colorFieldCombo = QtGui.QComboBox()
        colorFieldCombo.addItems([str(ii) for ii in range(1, 6)])
        morphFieldLabel = QtGui.QLabel(self.tr('Map Size to Field #'))
        morphFieldCombo = QtGui.QComboBox()
        morphFieldCombo.addItems([str(ii) for ii in range(1, 6)])

        layout.addWidget(sep, currentRow, 0, 1, -1)
        currentRow += 1
        layout.addWidget(colorFieldLabel, currentRow, 0, 1, 2)
        layout.addWidget(colorFieldCombo, currentRow, 2, 1, 2)
        currentRow += 1
        layout.addWidget(morphFieldLabel, currentRow, 0, 1, 2)
        layout.addWidget(morphFieldCombo, currentRow, 2, 1, 2)
        
        frame.setLayout(layout)

        gridButton = QtGui.QCheckBox('Enable Grid')
        
        layout = QtGui.QGridLayout()
        layout.addWidget(tree, 0, 0, 5, 5)
        layout.addWidget(frame, 0, 5, 1, 3)
        layout.addWidget(gridButton, 5, 0)
        self.setLayout(layout)
        for key, value in valueFieldMap.items():
            self.registerField('fieldName%d' % (key), value[1])
            self.registerField('valueMin%d' % (key), value[2])
            self.registerField('valueMax%d' % (key), value[3])
            self.setField(QtCore.QString('fieldName%d' % (key)), QtCore.QVariant(value[1].text()))
            self.setField(QtCore.QString('valueMin%d' % (key)), QtCore.QVariant(value[2].text()))
            self.setField(QtCore.QString('valueMax%d' % (key)), QtCore.QVariant(value[3].text()))
	self.registerField('wildcard', wildCardLineEdit)
        self.registerField('colorField', colorFieldCombo, 'currentText')
        self.registerField('morphField', morphFieldCombo, 'currentText')
        self.registerField('grid', gridButton)
	self.setField(QtCore.QString('wildcard'), QtCore.QVariant(wildCardLineEdit.text()))
        self.setField(QtCore.QString('colorField'), QtCore.QVariant(colorFieldCombo.currentText()))
        self.setField(QtCore.QString('morphField'), QtCore.QVariant(morphFieldCombo.currentText()))
        self.setField(QtCore.QString('grid'), QtCore.QVariant(gridButton.isChecked()))

    def nextId(self):
        return -1

    def _setTargetObject(self, item, column):
        self.setField(QtCore.QString('targetObject'), QtCore.QVariant(QtCore.QString(item.getMooseObject().path)))
        self.wizard()._targetObjectPath = item.getMooseObject().path

# from glcellloader import GLCellLoader
if __name__ == '__main__':
    app = QtGui.QApplication([])    
    widget = MooseGLWizard()
    # loader = GLCellLoader(
    widget.show()
    app.exec_()
    

# 
# glwizard.py ends here
