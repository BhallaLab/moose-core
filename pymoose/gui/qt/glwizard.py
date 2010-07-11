# glwizard.py --- 
# 
# Filename: glwizard.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Fri Jul  9 21:23:39 2010 (+0530)
# Version: 
# Last-Updated: Sun Jul 11 17:44:19 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 742
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
        self.initSettings()
        self._port = MooseGLWizard.currentPort
        self._targetObject = moose.Neutral('/')
        MooseGLWizard.currentPort += 1
        glclientPage = self._makeGLClientPage()
        self.setPage(MooseGLWizard.pageIdMap['glclient'], glclientPage)
        self.setPage(MooseGLWizard.pageIdMap['glcell'], self._makeGLCellPage())
        self.setPage(MooseGLWizard.pageIdMap['glview'], self._makeGLViewPage())
        self.accepted.connect(self.createGLSlot)
        

    def initSettings(self):
        self._settings = config.get_settings()
        self._colormap = unicode(self._settings.value(config.KEY_GL_COLORMAP).toString())
        
        if not self._colormap:
            self._colormap = config.GL_DEFAULT_COLORMAP
        if not os.path.isfile(self._colormap):
            self._colormap = unicode(QtGui.QFileDialog.getOpenFileName(self, 'Select Colormap File'))
        self._settings.setValue(config.KEY_GL_COLORMAP, self._colormap)
        self._glClientExe = unicode(self._settings.value(config.KEY_GL_CLIENT_EXECUTABLE).toString())
        if not self._glClientExe:
            self._glClientExe = config.GL_CLIENT_EXECUTABLE
        if not os.access(self._glClientExe, os.X_OK):
            self._glClientExe = QtGui.QFileDialog.getOpenFileName(self, 'Select GLClient Executable')
        if not os.access(self._glClientExe, os.X_OK):
            QtGui.QMessageBox.critical(self, 'No execute permission', 'Please select the correct glclient executable. Make sure you have execution permission on the file.')
        else:
            self._settings.setValue(config.KEY_GL_CLIENT_EXECUTABLE, self._glClientExe)
    
                                     
    def _makeGLClientPage(self):
        page = QtGui.QWizardPage(self)
        page.nextId = lambda glClientPage: MooseGLWizard.pageIdMap['glcell'] if glClientPage.field('glcellMode').toBool() else MooseGLWizard.pageIdMap['glview']


        page.setTitle(self.tr('Configure Client'))
        page.setSubTitle(self.tr('Setup the client for 3-D visualization'))
        
        exeLabel = QtGui.QLabel(self.tr('&Executable of glclient:'))
        exeLineEdit = QtGui.QLineEdit(self.tr(self._glClientExe))
        exeButton = QtGui.QPushButton(self.tr('Browse'))
        exeButton.clicked.connect(self._selectExeSlot)
        exeLabel.setBuddy(exeButton)

        colormapLabel = QtGui.QLabel(self.tr('&Colormap File'))
        colormapLineEdit = QtGui.QLineEdit(self.tr(self._colormap))
        colormapButton = QtGui.QPushButton(self.tr('Browse'))
        colormapButton.clicked.connect(self._selectColormapSlot)
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
        portLineEdit = QtGui.QLineEdit(str(self._port))
        portLabel.setBuddy(portLineEdit)
        
        syncButton = QtGui.QCheckBox(self.tr('&Sync'))
        syncButton.setToolTip(self.tr('Slow down the simulation to keep pace with the 3-D rendering.'))
        syncButton.setChecked(False)

        bgColorLabel = QtGui.QLabel(self.tr('&Background Color'))
        bgColorButton = QtGui.QPushButton('')
        bgColorButton.setObjectName('bgColorButton')
        black = QtGui.QColor(Qt.black)
        styleStr = QtCore.QString('QPushButton#bgColorButton {background-color: %s}' % black.name())
        bgColorButton.setStyleSheet(styleStr)
        bgColorButton.clicked.connect(self._chooseBackgroundColorSlot)
        bgColorLabel.setBuddy(bgColorButton)

        layout = QtGui.QGridLayout(page)
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

        page.registerField('glclient', exeLineEdit)
        page.registerField('colormap', colormapLineEdit)
        page.registerField('port', portLineEdit)
        page.registerField('glcellMode', glCellRadioButton)
        page.registerField('bgColorButton', bgColorButton)
        page.registerField('syncButton', syncButton)
        self._exeLineEdit = exeLineEdit
        self._colormapLineEdit = colormapLineEdit
        self._bgColorButton = bgColorButton
        return page

    def _selectExeSlot(self):
        self._glClientExe = unicode(QtGui.QFileDialog.getOpenFileName(self, 'Select glclient executable'))
        self._exeLineEdit.setText(self._glClientExe)

    def _selectColormapSlot(self):
        self._colormap = unicode(QtGui.QFileDialog.getOpenFileName(self, 'Select Colormap File'))
        self._colormapLineEdit.setText(self._colormap)

    def _makeGLCellPage(self):
        page = QtGui.QWizardPage()
        page.nextId = lambda glCellPage: -1
        page.setTitle('Select GLCell Target ')
        page.setSubTitle('Expand the model tree and select an element to observe in 3-D')
        layout = QtGui.QGridLayout()
        tree = MooseTreeWidget()
        tree.itemClicked.connect(self._setTargetObject)

        fieldsLabel = QtGui.QLabel(self.tr('&Field to observe'))
        fieldsLineEdit = QtGui.QLineEdit('Vm')
        fieldsLabel.setBuddy(fieldsLineEdit)

        frame = QtGui.QFrame()
        frame.setLayout(QtGui.QHBoxLayout())
        frame.setFrameStyle(QtGui.QFrame.Raised | QtGui.QFrame.StyledPanel)
        frame.layout().addWidget(fieldsLabel)
        frame.layout().addWidget(fieldsLineEdit)

        thresholdLabel = QtGui.QLabel(self.tr('&Threshold'))
        thresholdLineEdit = QtGui.QLineEdit('1')
        thresholdLabel.setBuddy(thresholdLineEdit)

        highValueLabel = QtGui.QLabel(self.tr('&High Value'))
        highValueLineEdit = QtGui.QLineEdit('')
        highValueLabel.setBuddy(highValueLineEdit)

        lowValueLabel = QtGui.QLabel(self.tr('&Low Value'))
        lowValueLineEdit = QtGui.QLineEdit('')
        lowValueLabel.setBuddy(lowValueLineEdit)

        vscaleLabel = QtGui.QLabel(self.tr('&Vscale'))
        vscaleLineEdit = QtGui.QLineEdit(self.tr('1.0'))
        vscaleLineEdit.setToolTip(self.tr('Scale up the thickness of compartments for visualization.'))
        vscaleLabel.setBuddy(vscaleLineEdit)
        
        layout.addWidget(tree, 0, 0, 5, 5)
        layout.addWidget(frame, 4, 5, 1, 3)
        layout.addWidget(thresholdLabel, 5, 0)
        layout.addWidget(thresholdLineEdit, 5, 1)
        layout.addWidget(vscaleLabel, 5, 3)
        layout.addWidget(vscaleLineEdit, 5, 4)
        layout.addWidget(lowValueLabel, 6, 1)
        layout.addWidget(lowValueLineEdit, 6, 2)
        layout.addWidget(highValueLabel, 6, 4)
        layout.addWidget(highValueLineEdit, 6, 5)
        
        page.setLayout(layout)
        page.registerField('glcellfield', fieldsLineEdit)
        page.registerField('threshold', thresholdLineEdit)
        page.registerField('vscale', vscaleLineEdit)
        page.registerField('highValue', highValueLineEdit)
        page.registerField('lowValue', lowValueLineEdit)

        self._fieldsLineEdit = fieldsLineEdit
        return page

    def _makeGLViewPage(self):
        page = QtGui.QWizardPage()
        page.nextId = lambda glViewPage: -1
        page.setTitle('Select GLView Target ')
        page.setSubTitle('Expand the model tree and select an element to observe in 3-D')
        tree = MooseTreeWidget()
        tree.itemClicked.connect(self._setTargetObject)

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
            valueFieldMap[ii] = (valueLabel, fieldNameEdit, valueMinEdit, valueMaxEdit)
            layout.addWidget(valueLabel, currentRow, 0)
            layout.addWidget(fieldNameEdit, currentRow, 1)
            layout.addWidget(valueMinEdit, currentRow, 2)
            layout.addWidget(valueMaxEdit, currentRow, 3)                    
            currentRow += 1
        sep = QtGui.QFrame()
        sep.setFrameStyle(QtGui.QFrame.Sunken | QtGui.QFrame.VLine)
        colorFieldLabel = QtGui.QLabel('Map Color to Field #')
        colorFieldCombo = QtGui.QComboBox()
        colorFieldCombo.addItems([str(ii) for ii in range(1, 6)])
        morphFieldLabel = QtGui.QLabel('Map Size to Field #')
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
        page.setLayout(layout)
        for key, value in valueFieldMap.items():
            page.registerField('fieldName%d' % (key), value[1])
            page.registerField('valueMin%d' % (key), value[2])
            page.registerField('valueMax%d' % (key), value[3])
        page.registerField('colorField', colorFieldCombo, 'currentText')
        page.registerField('morphField', morphFieldCombo, 'currentText')
        page.registerField('grid', gridButton)
        return page

    def _setTargetObject(self, item, column):
        self._targetObject = item.getMooseObject()

    def _chooseBackgroundColorSlot(self):
        currentColor = self._bgColorButton.palette().color(QtGui.QPalette.Background)
        color = QtGui.QColorDialog.getColor(currentColor, self)
        if color.isValid():
            style = QtCore.QString('QPushButton#bgColorButton {Background-color: %s}' % color.name())
            self._bgColorButton.setStyleSheet(style)
            self._bgColorButton.setText(color.name())

    def createGLSlot(self):
        bgColor = self._bgColorButton.palette().color(QtGui.QPalette.Background).name()
        bgColor = bgColor[1:]
        sync = 'on' if self.field('syncButton').toBool() else 'off'
        glcellMode = self.field('glcellMode').toBool()
        port = str(self.field('port').toString())
        glclient = str(self.field('glclient').toString())
        colormap = str(self.field('colormap').toString())
        print '%s\n%s\n%d\n%s\n%s\n%s' % (bgColor, sync, glcellMode, port, glclient, colormap)
        client = self._mooseHandler.startGLClient(glclient, port, 'c' if glcellMode else 'v', colormap)
        time.sleep(3)
        if glcellMode:
            try:
                threshold = self.field('threshold').toDouble()
            except ValueError:
                threshold = None
            try:
                highValue = self.field('highValue').toDouble()
            except ValueError:
                highValue = None
            try:
                lowValue = self.field('lowValue').toDouble()
            except ValueError:
                lowValue = None
            try:
                vscale = self.field('vscale').toDouble()
            except:
                vscale = None
            
            field = str(self.field('glcellfield').toString())
                
            self._mooseHandler.makeGLCell(self._targetObject.path, port, field, threshold, lowValue, highValue, vscale, bgColor, sync)
        else:
            field = []
            valueMin = []
            valueMax = []
            for ii in range(5):
                field.append(str(self.field('fieldName%d' % (ii+1)).toString()))
                valueMin.append(self.field('valueMin%d' % (ii+1)).toDouble())
                valueMax.append(self.field('valueMax%d' % (ii+1)).toDouble())
            colorFieldIndex = self.field('colorField').toInt()
            morphFieldIndex = self.field('morphField').toInt()
            grid = 'on' if self.field('grid').toBool() else 'off'
            self._mooseHandler.makeGLView(self._targetObject.path, self._port, field, valueMin, valueMax, colorFieldIndex, morphFieldIndex, grid, bgColor, sync)

# from glcellloader import GLCellLoader
if __name__ == '__main__':
    app = QtGui.QApplication([])    
    widget = MooseGLWizard()
    # loader = GLCellLoader(
    widget.show()
    app.exec_()
    

# 
# glwizard.py ends here
