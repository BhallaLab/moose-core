#!/usr/bin/env python
# moosegui.py --- 
# 
# Filename: moosegui.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Wed Jan 20 15:24:05 2010 (+0530)
# Version: 
# Last-Updated: Mon Apr 18 15:05:30 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 2653
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

from __future__ import with_statement    

import os
import sys
import code
import subprocess
from datetime import date
from collections import defaultdict


python_version = sys.version_info
required_version = (2,5)
if  python_version[0] < required_version[0] or python_version[0] == required_version[0] and python_version[1] < required_version[1]:
    raise 'Need Python version 2.5 or greater'

from PyQt4 import QtCore, QtGui
from PyQt4.Qt import Qt

import config
from glclientgui import GLClientGUI
# The following line is for ease in development environment. Normal
# users will have moose.py and _moose.so installed in some directory
# in PYTHONPATH.  If you have these two files in /usr/local/moose, you
# can enter the command:
#
# "export PYTHONPATH=$PYTHONPATH:/usr/local/moose" 
#
# in the command prompt before running the
# moosegui with "python moosegui.py"
# sys.path.append('/home/subha/src/moose/pymoose')



# These are the MOOSE GUI specific imports
from objectedit import ObjectFieldsModel, ObjectEditDelegate, ObjectEditView
from moosetree import *
from mooseclasses import *
from mooseglobals import MooseGlobals
from mooseshell import MooseShell
from moosehandler import MooseHandler
from mooseplot import MoosePlot, MoosePlotWindow
from plotconfig import PlotConfig
from glwizard import MooseGLWizard
from firsttime import FirstTimeWizard
#from layout import Screen
import layout
from updatepaintGL import *
from vizParasDialogue import *


def makeClassList(parent=None, mode=MooseGlobals.MODE_ADVANCED):
    """Make a list of classes that can be used in current mode
    mode can be all, kinetikit, neurokit.
    In all mode, all classes are shown.
    In kinetikit mode only chemical kinetics classes are shown.
    In neurokit mode, only neuronal simulation classes are shown.
    """
    if mode == MooseGlobals.MODE_ADVANCED:
	return MooseClassWidget(parent)
    elif mode == MooseGlobals.MODE_KKIT:
        pass
    elif mode == MooseGlobals.MODE_NKIT:
        pass
    else:
	print 'Error: makeClassList() - mode:', mode, 'is undefined.'

class MainWindow(QtGui.QMainWindow):
    default_plot_count = 1
    def __init__(self, interpreter=None, parent=None):
	QtGui.QMainWindow.__init__(self, parent)
        self.settingsReset = False
        self.setWindowTitle('MOOSE GUI')
        self.mooseHandler = MooseHandler()
        self.connect(self.mooseHandler, QtCore.SIGNAL('updatePlots(float)'), self.updatePlots)
        self.settings = config.get_settings()        
        self.demosDir = str(self.settings.value(config.KEY_DEMOS_DIR).toString())
        # print self.demosDir
        # if not self.demosDir:
        #     self.demosDir = str(QtGui.QFileDialog.getExistingDirectory(self, 'Please select pymoose demos directory'))
        self.resize(800, 600)
        self.setDockOptions(self.AllowNestedDocks | self.AllowTabbedDocks | self.ForceTabbedDocks | self.AnimatedDocks)        
        self.setDockNestingEnabled(True)
        
        #add_chait
        self.mainCentralWidget = QtGui.QWidget(self)
        self.horizontalLayout = QtGui.QHBoxLayout(self.mainCentralWidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        #add_chait
        self.centralVizPanel = QtGui.QMdiArea(self)
        self.horizontalLayout.addWidget(self.centralVizPanel)
        self.centralVizPanel.setViewMode(self.centralVizPanel.TabbedView)

        self.centralPanel = QtGui.QMdiArea(self)
        self.horizontalLayout.addWidget(self.centralPanel)
        
        # The following are for holding transient selections from
        # connection dialog
        self._srcElement = None
        self._destElement = None
        self._srcField = None
        self._destField = None
        
        # plots is a list of available MoosePlot widgets.
        self.plots = []
        self.plotWindows = []
        
        self.vizs = []		#add_chait
        self.vizWindows = []	#add_chait
        
        # We use the objFieldEditorMap to store reference to cache the
        # model objects for moose objects.
        self.objFieldEditorMap = {}
        # This is the element tree of MOOSE
        self.createMooseTreePanel()
        # List of classes - one can double click on any class to
        # create an instance under the currently selected element in
        # mooseTreePanel
        
        self.createMooseClassesPanel()

        # Create a widget to configure the glclient
        # self.createGLClientDock()
        self.createControlDock()
        # Connect the double-click event on modelTreeWidget items to
        # creation of the object editor.
        self.connect(self.modelTreeWidget, 
                     QtCore.SIGNAL('itemDoubleClicked(QTreeWidgetItem *, int)'),
                     self.makeObjEditorFromTreeItem)
        self.connect(self.modelTreeWidget,
                     QtCore.SIGNAL('mooseObjectInserted(PyQt_PyObject)'),
                     self.makeObjectFieldEditor)
        self.connect(self.modelTreeWidget, 
                     QtCore.SIGNAL('itemClicked(QTreeWidgetItem *, int)'),
                     self.setCurrentElement)
        
        self.makeShellDock(interpreter)
	
        # By default, we show information about MOOSE in the central widget
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        
        # tablePlotMap is a maps all currently available tables to the
        # plot widgets they belong to.
        self.tablePlotMap = {}
        self.currentPlotWindow = None
        # Start with a default number of plot widgets
        self._visiblePlotWindowCount = 0
        for ii in range(MainWindow.default_plot_count):
            self.addPlotWindow()
        self.plotConfig = PlotConfig(self)
        self.plotConfig.setVisible(False)        
        #add_chait
        #self.setCentralWidget(self.centralPanel)
        self.setCentralWidget(self.mainCentralWidget)
        
        self.centralPanel.tileSubWindows()
        self.connect(self.centralPanel, QtCore.SIGNAL('subWindowActivated(QMdiSubWindow *)'), self.setCurrentPlotWindow)

	# We connect the double-click event on the class-list to
        # insertion of moose object in model tree.
        self.connect(self.mooseClassesWidget, 
                     QtCore.SIGNAL('classNameDoubleClicked(PyQt_PyObject)'), 
                     self.modelTreeWidget.insertMooseObjectSlot)
        self.connect(QtGui.qApp, QtCore.SIGNAL('lastWindowClosed()'), self.saveLayout)
        self.createActions()
        self.makeMenu()
        # Loading of layout should happen only after all dockwidgets
        # have been created
        #add_chait
        self.createSimulationToolbar()
        self.loadLayout()

    def createSimulationToolbar(self):
        self.simToolbar = QtGui.QToolBar(self)
        self.simToolbar.setObjectName('simToolbar')
        self.simToolbar.setFloatable(False)
        self.simToolbar.setMovable(False)

        
        self.runTimeLabelToolbar = QtGui.QLabel(self.simToolbar)
        self.runTimeLabelToolbar.setText('Run Time(secs):')
        self.runTimeLabelToolbar.setGeometry(10,0,120,30)

        self.runTimeEditToolbar = QtGui.QLineEdit(self.simToolbar)
        self.runTimeEditToolbar.setText('%1.3e' % (MooseHandler.runtime))
        self.runTimeEditToolbar.setGeometry(130,0,100,30)
        
        self.currentTimeLabelToolbar = QtGui.QLabel(self.simToolbar)
        self.currentTimeLabelToolbar.setText(' Current Time :')
        self.currentTimeLabelToolbar.setGeometry(230,0,120,30)

        self.currentTimeEditToolbar = QtGui.QLabel(self.simToolbar)
        self.currentTimeEditToolbar.setText('0')
        self.currentTimeEditToolbar.setGeometry(350,0,120,30)
        
        self.runButtonToolbar = QtGui.QToolButton(self.simToolbar)
        self.runButtonToolbar.setText('Run')
        self.runButtonToolbar.setGeometry(470,0,100,30)
#        print self.runButton.size()
#        self.simToolbar.addSeparator()

        self.continueButtonToolbar = QtGui.QToolButton(self.simToolbar)
        self.continueButtonToolbar.setText('Continue')
        self.continueButtonToolbar.setGeometry(580,0,100,30)


        self.connect(self.runButtonToolbar, QtCore.SIGNAL('clicked()'), self.resetAndRunSlot1)
        self.connect(self.continueButtonToolbar, QtCore.SIGNAL('clicked()'), self._runSlot1)

        self.simToolbar.show()
        self.simToolbar.setMinimumHeight(30)
#        print self.simToolbar.size()
        self.addToolBar(Qt.TopToolBarArea,self.simToolbar)

    
    def makeConnectionPopup(self):
        """Create a dialog to connect moose objects via messages."""
        self.connectionDialog = QtGui.QDialog(self)
        self.connectionDialog.setWindowTitle(self.tr('Create Connection'))
        self.connectionDialog.setModal(False)

        self.sourceTree = MooseTreeWidget(self.connectionDialog)
        self.destTree = MooseTreeWidget(self.connectionDialog)
        self.connect(self.sourceTree, QtCore.SIGNAL('itemClicked(QTreeWidgetItem *, int)'), self.selectConnSource)
        self.connect(self.destTree, QtCore.SIGNAL('itemClicked(QTreeWidgetItem*, int)'), self.selectConnDest)
        sourceObjLabel = QtGui.QLabel(self.tr('Selected Source'))
        sourceFieldLabel = QtGui.QLabel(self.tr('Source Field'), self.connectionDialog)
        self.sourceFieldComboBox = QtGui.QComboBox(self.connectionDialog)

        destObjLabel = QtGui.QLabel(self.tr('Selected Destination'))        
        destFieldLabel = QtGui.QLabel(self.tr('Target Field'), self.connectionDialog)
        self.destFieldComboBox = QtGui.QComboBox(self.connectionDialog)

        self.sourceObjText = QtGui.QLineEdit(self.connectionDialog)
        self.destObjText = QtGui.QLineEdit(self.connectionDialog)

        okButton = QtGui.QPushButton(self.tr('OK'), self.connectionDialog)
        self.connect(okButton, QtCore.SIGNAL('clicked()'), self.connectionDialog.accept)
        cancelButton = QtGui.QPushButton(self.tr('Cancel'), self.connectionDialog)
        self.connect(cancelButton, QtCore.SIGNAL('clicked()'), self.connectionDialog.reject)

        self.connect(self.connectionDialog, QtCore.SIGNAL('accepted()'), self.createConnection)
        self.connect(self.connectionDialog, QtCore.SIGNAL('rejected()'), self.cancelConnection)

        layout = QtGui.QGridLayout()
        layout.addWidget(self.sourceTree, 0, 0, 5, 2)
        layout.addWidget(sourceObjLabel, 6, 0)
        layout.addWidget(self.sourceObjText, 6, 1, 1, 2)
        layout.addWidget(sourceFieldLabel, 7, 0)
        layout.addWidget(self.sourceFieldComboBox, 7, 1)
        sep = QtGui.QFrame(self.connectionDialog)
        sep.setFrameStyle(QtGui.QFrame.VLine | QtGui.QFrame.Sunken)
        layout.addWidget(sep, 0, 3, -1, 1)
        layout.addWidget(self.destTree, 0, 4, 5, 2)
        layout.addWidget(destObjLabel, 6, 4)
        layout.addWidget(self.destObjText, 6, 5, 1, 2)
        layout.addWidget(destFieldLabel, 7, 4)
        layout.addWidget(self.destFieldComboBox, 7, 5)
        
        layout.addWidget(cancelButton, 9, 0)
        layout.addWidget(okButton, 9, 4)

        self.connectionDialog.setLayout(layout)
        self.connectionDialog.show()

    def cancelConnection(self):
        self._connSrcObject = None
        self._connDestObject = None

    def makeAboutMooseLabel(self):
            """Create a QLabel with basic info about MOOSE."""
            sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
            aboutText = '<html><h3 align="center">%s</h3><p>%s</p><p>%s</p><p>%s</p><p align="center">Home Page: <a href="%s">%s</a></p></html>' % \
                (MooseGlobals.TITLE_TEXT, 
                 MooseGlobals.COPYRIGHT_TEXT, 
                 MooseGlobals.LICENSE_TEXT, 
                 MooseGlobals.ABOUT_TEXT,
                 MooseGlobals.WEBSITE,
                 MooseGlobals.WEBSITE)
            aboutMooseMessage = QtGui.QMessageBox.about(self, self.tr('About MOOSE'), self.tr(aboutText))
            return aboutMooseMessage

    def showRightBottomDocks(self, checked):
        """Hides the widgets on right and bottom dock area"""
        for child in self.findChildren(QtGui.QDockWidget):
            area = self.dockWidgetArea(child)
            if ( area == QtCore.Qt.BottomDockWidgetArea) or \
                    (area == QtCore.Qt.RightDockWidgetArea):
                child.setVisible(checked)


    def makeShellDock(self, interpreter=None, mode=MooseGlobals.CMD_MODE_PYMOOSE):
        """A MOOSE command line for GENESIS/Python interaction"""
        self.commandLineDock = QtGui.QDockWidget(self.tr('MOOSE Shell'), self)
        self.commandLineDock.setObjectName(self.tr('MooseShell'))
        self.shellWidget = MooseShell(interpreter, message='MOOSE Version %s' % (moose.version))
        self.shellWidget.setFrameStyle(QtGui.QFrame.Raised | QtGui.QFrame.StyledPanel)
        self.commandLineDock.setWidget(self.shellWidget)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.commandLineDock)
        self.commandLineDock.setObjectName('MooseCommandLine')
        #add_chait
        self.commandLineDock.setMaximumHeight(180)
        return self.commandLineDock


    def makeObjEditorFromTreeItem(self, item, column):
        """Wraps makeObjectFieldEditor for use via a tree item"""
        obj = item.getMooseObject()
        self.makeObjectFieldEditor(obj)

    def makeObjectFieldEditor(self, obj):
        """Creates a table-editor for a selected object."""
        if obj.className == 'Shell' or obj.className == 'PyMooseContext' or obj.className == 'GenesisParser':
            print '%s of class %s is a system object and not to be edited in object editor.' % (obj.path, obj.className)
            return
        try:
            self.objFieldEditModel = self.objFieldEditorMap[obj.id]
            config.LOGGER.debug('Found model %s for object %s' % (self.objFieldEditModel, obj.id))
        except KeyError:
            config.LOGGER.debug('No editor for this object: %s' % (obj.path))
            self.objFieldEditModel = ObjectFieldsModel(obj)
            self.objFieldEditorMap[obj.id] = self.objFieldEditModel
            self.connect(self.objFieldEditModel, QtCore.SIGNAL('plotWindowChanged(const QString&, const QString&)'), self.changeFieldPlotWidget)
            
            if  not hasattr(self, 'objFieldEditPanel'):
                self.objFieldEditPanel = QtGui.QDockWidget(self.tr(obj.name), self)
                self.objFieldEditPanel.setObjectName(self.tr('MooseObjectFieldEdit'))
                #if (config.QT_MAJOR_VERSION > 4) or ((config.QT_MAJOR_VERSION == 4) and (config.QT_MINOR_VERSION >= 5)):
                #    self.tabifyDockWidget(self.mooseClassesPanel, self.objFieldEditPanel)
                #else:
                #    self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.objFieldEditPanel)
                #add_chait
                self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.objFieldEditPanel)
                self.restoreDockWidget(self.objFieldEditPanel)
            self.connect(self.objFieldEditModel, QtCore.SIGNAL('objectNameChanged(PyQt_PyObject)'), self.renameObjFieldEditPanel)
            
                
        self.objFieldEditor = ObjectEditView(self.objFieldEditPanel)
        self.objFieldEditor.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.objFieldEditor.setObjectName(str(obj.id)) # Assuming Id will not change in the lifetime of the object - something that might break in future version.
        self.objFieldEditor.setModel(self.objFieldEditModel)
        config.LOGGER.debug('Set model to: %s' % (str(self.objFieldEditModel)))
        self.objFieldEditor.setEditTriggers(QtGui.QAbstractItemView.DoubleClicked
                                 | QtGui.QAbstractItemView.SelectedClicked)
        self.objFieldEditor.setDragEnabled(True)
        for plot in self.plots:
            objName = plot.objectName()
            if objName not in self.objFieldEditModel.plotNames :
                self.objFieldEditModel.plotNames += [plot.objectName() for plot in self.plots]
        self.objFieldEditor.setItemDelegate(ObjectEditDelegate(self))
        self.connect(self.objFieldEditModel, 
                     QtCore.SIGNAL('objectNameChanged(PyQt_PyObject)'),
                     self.modelTreeWidget.updateItemSlot)
        if hasattr(self, 'sceneLayout'):
	        self.connect(self.objFieldEditModel, 
        	             QtCore.SIGNAL('objectNameChanged(PyQt_PyObject)'),
        	             self.sceneLayout.updateItemSlot)

        self.objFieldEditPanel.setWidget(self.objFieldEditor)
        self.objFieldEditPanel.setWindowTitle(self.tr(obj.name))
        self.objFieldEditPanel.raise_()
	self.objFieldEditPanel.show()
        #add_chait
        self.objFieldEditPanel.setMinimumWidth(300)

    def createGLCellWidget(self):
    	"""Create a GLCell object to show the currently selected cell"""
        raise DeprecationWarning('This function is not implemented properly and is deprecated.')
        cellItem = self.modelTreeWidget.currentItem()
        cell = cellItem.getMooseObject()
        if not cell.className == 'Cell':
            QtGui.QMessageBox.information(self, self.tr('Incorrect type for GLCell'), 
                                          self.tr('GLCell is for visualizing a cell. Please select one in the Tree view. Currently selected item is of ' 
                                                  + cell.className 
                                                  + ' class. Hover mouse over an item to see its class.'))
            return
        
    def createActions(self):
        # Actions for view menu
        # The following actions are to toggle visibility of various widgets
        self.controlDockAction = self.controlDock.toggleViewAction()
        # self.glClientAction = self.glClientDock.toggleViewAction()
        # self.glClientAction.setChecked(False)
        self.mooseTreeAction = self.mooseTreePanel.toggleViewAction()
        self.refreshMooseTreeAction = QtGui.QAction(self.tr('Refresh model tree'), self)
   	self.connect(self.refreshMooseTreeAction, QtCore.SIGNAL('triggered(bool)'), self.modelTreeWidget.recreateTree)
        self.mooseClassesAction = self.mooseClassesPanel.toggleViewAction()
        self.mooseShellAction = self.commandLineDock.toggleViewAction()
        self.mooseShellAction.setChecked(False)
        self.mooseGLCellAction = QtGui.QAction(self.tr('GLCell'), self)
        self.mooseGLCellAction.setChecked(False)
        self.connect(self.mooseGLCellAction, QtCore.SIGNAL('triggered()'), self.createGLCellWidget)

        self.autoHideAction = QtGui.QAction(self.tr('Autohide during simulation'), self)
	self.autoHideAction.setCheckable(True)
        self.autoHideAction.setChecked(config.get_settings().value(config.KEY_RUNTIME_AUTOHIDE).toBool())

        self.showRightBottomDocksAction = QtGui.QAction(self.tr('Right and Bottom Docks'), self)
	self.showRightBottomDocksAction.setCheckable(True)
	self.connect(self.showRightBottomDocksAction, QtCore.SIGNAL('triggered(bool)'), self.showRightBottomDocks)
        self.showRightBottomDocksAction.setChecked(False)

        self.tabbedViewAction = QtGui.QAction(self.tr('Tabbed view'), self)
        self.tabbedViewAction.setCheckable(True)
        self.tabbedViewAction.setChecked(False)
        self.connect(self.tabbedViewAction, QtCore.SIGNAL('triggered(bool)'), self.switchTabbedView)

        self.tilePlotWindowsAction = QtGui.QAction(self.tr('Tile Plots'), self)
	self.tilePlotWindowsAction.setCheckable(True)
	self.connect(self.tilePlotWindowsAction, QtCore.SIGNAL('triggered(bool)'), self.centralPanel.tileSubWindows)
        self.cascadePlotWindowsAction = QtGui.QAction(self.tr('Cascade Plots'), self)
        self.cascadePlotWindowsAction.setCheckable(True)
	self.connect(self.cascadePlotWindowsAction, QtCore.SIGNAL('triggered(bool)'), self.centralPanel.cascadeSubWindows)
        
        self.subWindowLayoutActionGroup = QtGui.QActionGroup(self)
        self.subWindowLayoutActionGroup.addAction(self.tilePlotWindowsAction)
        self.subWindowLayoutActionGroup.addAction(self.cascadePlotWindowsAction)
        self.subWindowLayoutActionGroup.setExclusive(True)
        self.tilePlotWindowsAction.setChecked(True)
        self.togglePlotWindowsAction = QtGui.QAction(self.tr('Plot windows'), self)
        self.togglePlotWindowsAction.setCheckable(True)
        self.togglePlotWindowsAction.setChecked(True)
        self.connect(self.togglePlotWindowsAction, QtCore.SIGNAL('triggered(bool)'), self.setPlotWindowsVisible)

        # Action to configure plots
        self.configurePlotAction = QtGui.QAction(self.tr('Configure selected plots'), self)
	self.connect(self.configurePlotAction, QtCore.SIGNAL('triggered(bool)'), self.configurePlots)
        self.togglePlotVisibilityAction = QtGui.QAction(self.tr('Hide selected plots'), self)
        self.togglePlotVisibilityAction.setCheckable(True)
        self.togglePlotVisibilityAction.setChecked(False)
        self.connect(self.togglePlotVisibilityAction, QtCore.SIGNAL('triggered(bool)'), self.togglePlotVisibility)
        self.showAllPlotsAction = QtGui.QAction(self.tr('Show all plots'), self)
        self.connect(self.showAllPlotsAction, QtCore.SIGNAL('triggered()'), self.showAllPlots)
        

        # Action to create connections
        self.connectionDialogAction = QtGui.QAction(self.tr('&Connect elements'), self)
	self.connect(self.connectionDialogAction, QtCore.SIGNAL('triggered()'), self.makeConnectionPopup)

        # Actions for file menu
        self.loadModelAction = QtGui.QAction(self.tr('Load Model'), self)
        self.loadModelAction.setShortcut(QtGui.QKeySequence(self.tr('Ctrl+L')))
        self.connect(self.loadModelAction, QtCore.SIGNAL('triggered()'), self.popupLoadModelDialog)
        
        self.newGLWindowAction = QtGui.QAction(self.tr('New GL Window'), self) #add_chait
        self.connect(self.newGLWindowAction, QtCore.SIGNAL('triggered(bool)'), self.addGLWindow)
        
        self.newPlotWindowAction = QtGui.QAction(self.tr('New Plot Window'), self)
        self.connect(self.newPlotWindowAction, QtCore.SIGNAL('triggered(bool)'), self.addPlotWindow)
        self.firstTimeWizardAction = QtGui.QAction(self.tr('FirstTime Configuration Wizard'), self)
        self.connect(self.firstTimeWizardAction, QtCore.SIGNAL('triggered(bool)'), self.startFirstTimeWizard)
        self.resetSettingsAction = QtGui.QAction(self.tr('Reset Settings'), self)
        self.connect(self.resetSettingsAction, QtCore.SIGNAL('triggered()'), self.resetSettings)

        # Actions to switch the command line between python and genesis mode.
        self.shellModeActionGroup = QtGui.QActionGroup(self)
        self.pythonModeAction = QtGui.QAction(self.tr('Python'), self.shellModeActionGroup)
	self.pythonModeAction.setCheckable(True)
        self.pythonModeAction.setChecked(True)
        self.genesisModeAction = QtGui.QAction(self.tr('GENESIS'), self.shellModeActionGroup)
	self.genesisModeAction.setCheckable(True)
        self.shellModeActionGroup.setExclusive(True)
        self.connect(self.shellModeActionGroup, QtCore.SIGNAL('triggered(QAction*)'), self.changeShellMode)
        
        # Quit action
        self.quitAction = QtGui.QAction(self.tr('&Quit'), self)
        self.quitAction.setShortcut(QtGui.QKeySequence(self.tr('Ctrl+Q')))
        self.connect(self.quitAction, QtCore.SIGNAL('triggered()'), self.doQuit)
	
        # Actions for 3D visualization
        self.startGLWizardAction = QtGui.QAction(self.tr('Start GL&Wizard'), self)
        self.connect(self.startGLWizardAction, QtCore.SIGNAL('triggered()'), self.startGLWizard)
        self.stopGLAction = QtGui.QAction(self.tr('Stop G&L Visualization'), self)
        self.connect(self.stopGLAction, QtCore.SIGNAL('triggered()'), self.mooseHandler.stopGL)

        # Help menu actions
        self.aboutMooseAction = QtGui.QAction(self.tr('&About'), self)
        self.connect(self.aboutMooseAction, QtCore.SIGNAL('triggered()'), self.makeAboutMooseLabel)
        self.showDocAction = QtGui.QAction(self.tr('Documentation'), self)
        self.reportBugAction = QtGui.QAction(self.tr('Report a bug'), self)
        self.connect(self.reportBugAction, QtCore.SIGNAL('triggered()'), self.openBugsPage)
        self.connect(self.showDocAction, QtCore.SIGNAL('triggered()'), self.browseDocumentation)
        # self.contextHelpAction = QtGui.QAction(self.tr('Context Help'), self)
        
        # Run menu action
        self.runAction = QtGui.QAction(self.tr('Run'), self)        
	self.connect(self.runAction, QtCore.SIGNAL('triggered(bool)'), self.resetAndRunSlot)
        # self.resetAction = QtGui.QAction(self.tr('Reset Simulation'), self)
	# self.connect(self.resetAction, QtCore.SIGNAL('triggered()'), self.resetSlot)
        self.continueRunAction = QtGui.QAction(self.tr('Continue'), self)
        self.connect(self.continueRunAction, QtCore.SIGNAL('triggered()'), self._runSlot)

    
    def runMichaelisMentenDemo(self):
        path = os.path.join(self.demosDir, 'michaelis_menten', 'guimichaelis.py')
        print 'Going to run:', path
        subprocess.call(['python', path])
        
        
    def runSquidDemo(self):
        path = os.path.join(self.demosDir, 'squid', 'qtSquid.py')
        print 'Going to run:', path
        subprocess.call(['python', path])

    def runIzhikevichDemo(self):
        path = os.path.join(self.demosDir, 'izhikevich', 'demogui_qt.py')
        print path
        subprocess.call(['python', path])

    def runGLCellDemo(self):
        path = os.path.join(self.demosDir, 'gl', 'glcelldemo.py')
        print path
        subprocess.call(['python', path], cwd=os.path.dirname(path))

    def runGLViewDemo(self):
        path = os.path.join(self.demosDir, 'gl', 'glviewdemo.py')
        print path
        subprocess.call(['python', path], cwd=os.path.dirname(path))
        
    
    def makeDemosMenu(self):
        self.MichaelisMentenDemoAction = QtGui.QAction(self.tr('Michaelis-Menten reaction'), self)
        self.connect(self.MichaelisMentenDemoAction, QtCore.SIGNAL('triggered()'), self.runMichaelisMentenDemo)
        self.squidDemoAction = QtGui.QAction(self.tr('Squid Axon'), self)
        self.connect(self.squidDemoAction, QtCore.SIGNAL('triggered()'), self.runSquidDemo)
        self.IzhikevichDemoAction = QtGui.QAction(self.tr('Izhikevich Model'), self)
        self.connect(self.IzhikevichDemoAction, QtCore.SIGNAL('triggered()'), self.runIzhikevichDemo)
        self.glCellDemoAction = QtGui.QAction(self.tr('GL Cell'), self)
        self.connect(self.glCellDemoAction, QtCore.SIGNAL('triggered(bool)'), self.runGLCellDemo)
        self.glViewDemoAction = QtGui.QAction(self.tr('GL View'), self)
        self.connect(self.glViewDemoAction, QtCore.SIGNAL('triggered(bool)'), self.runGLViewDemo)
        menu = QtGui.QMenu('&Demos and Tutorials', self)
        menu.addAction(self.MichaelisMentenDemoAction)
        menu.addAction(self.squidDemoAction)
        menu.addAction(self.IzhikevichDemoAction)
        menu.addAction(self.glCellDemoAction)
        menu.addAction(self.glViewDemoAction)
        return menu
        
    def makeMenu(self):
        self.fileMenu = QtGui.QMenu(self.tr('&File'), self)
        self.fileMenu.addAction(self.newPlotWindowAction)
        self.fileMenu.addAction(self.newGLWindowAction)	#add_chait
        self.fileMenu.addAction(self.loadModelAction)
        self.shellModeMenu = self.fileMenu.addMenu(self.tr('Moose Shell mode'))
        self.shellModeMenu.addActions(self.shellModeActionGroup.actions())
        self.fileMenu.addAction(self.firstTimeWizardAction)
        self.fileMenu.addAction(self.resetSettingsAction)
        self.fileMenu.addAction(self.quitAction)

        self.viewMenu = QtGui.QMenu('&View', self)
        self.viewMenu.addAction(self.controlDockAction)
        # self.viewMenu.addAction(self.glClientAction)
        self.viewMenu.addAction(self.mooseTreeAction)
        self.viewMenu.addAction(self.refreshMooseTreeAction)
        self.viewMenu.addAction(self.mooseClassesAction)
        self.viewMenu.addAction(self.mooseShellAction)
        self.viewMenu.addAction(self.autoHideAction)
        self.viewMenu.addAction(self.showRightBottomDocksAction)
        self.viewMenu.addSeparator().setText(self.tr('Layout Plot Windows'))
        self.viewMenu.addAction(self.tabbedViewAction)
        self.viewMenu.addAction(self.tilePlotWindowsAction)
        self.viewMenu.addAction(self.cascadePlotWindowsAction)
        self.viewMenu.addAction(self.togglePlotWindowsAction)
        self.runMenu = QtGui.QMenu(self.tr('&Run'), self)
        # self.runMenu.addAction(self.resetAction)
        self.runMenu.addAction(self.runAction)
        self.runMenu.addAction(self.continueRunAction)


        self.editModelMenu = QtGui.QMenu(self.tr('&Edit Model'), self)
        self.editModelMenu.addAction(self.connectionDialogAction)
        
		
        self.plotMenu = QtGui.QMenu(self.tr('&Plot Settings'), self)
        self.plotMenu.addAction(self.configurePlotAction)
        self.plotMenu.addAction(self.togglePlotVisibilityAction)
        self.plotMenu.addAction(self.showAllPlotsAction)
        #self.glMenu = QtGui.QMenu(self.tr('Open&GL'), self)
        #self.glMenu.addAction(self.startGLWizardAction)
        #self.glMenu.addAction(self.stopGLAction)
                                  
        self.helpMenu = QtGui.QMenu('&Help', self)
        self.helpMenu.addAction(self.showDocAction)
        # self.helpMenu.addAction(self.contextHelpAction)
        self.demosMenu = self.makeDemosMenu()
        self.helpMenu.addMenu(self.demosMenu)
        self.helpMenu.addAction(self.aboutMooseAction)
        self.helpMenu.addAction(self.reportBugAction)
        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.editModelMenu)
        self.menuBar().addMenu(self.runMenu)
        self.menuBar().addMenu(self.plotMenu)
        #self.menuBar().addMenu(self.glMenu)
        self.menuBar().addMenu(self.helpMenu)

    def createConnectionMenu(self, clickpoint):
        # index = self.modelTreeWidget.indexAt(clickpoint)
        mooseObject = self.modelTreeWidget.currentItem().getMooseObject()
        self.connectionMenu = QtGui.QMenu(self.tr('Connect'), self.modelTreeWidget)
        self.srcMenu = QtGui.QMenu(self.tr('Source Field'), self)
        self.connectionMenu.addMenu(self.srcMenu)
        self.destMenu = QtGui.QMenu(self.tr('Destination Field'), self)
        self.connectionMenu.addMenu(self.destMenu)
        self.connectActionGroup = QtGui.QActionGroup(self)
        srcFields = self.mooseHandler.getSrcFields(mooseObject)
        destFields = self.mooseHandler.getDestFields(mooseObject)
        for field in srcFields:
            action = QtGui.QAction(self.tr(field), self.connectActionGroup, triggered=self.setConnSrc)
            action.setData(QtCore.QVariant(QtCore.QString(mooseObject.path + '/' + field)))
            self.srcMenu.addAction(action)
        for field in destFields:
            action = QtGui.QAction(self.tr(field), self.connectActionGroup, triggered=self.setConnDest)
            action.setData(QtCore.QVariant(QtCore.QString(mooseObject.path + '/' + field)))
            self.destMenu.addAction(action)
        self.connectActionGroup.setExclusive(True)
        self.connectionMenu.popup(clickpoint)
        
    def saveLayout(self):
        '''Save window layout'''
        if self.settingsReset:
            return
        geo_data = self.saveGeometry()
        layout_data = self.saveState()
        config.get_settings().setValue(config.KEY_WINDOW_GEOMETRY, QtCore.QVariant(geo_data))
        config.get_settings().setValue(config.KEY_WINDOW_LAYOUT, QtCore.QVariant(layout_data))
        config.get_settings().setValue(config.KEY_RUNTIME_AUTOHIDE, QtCore.QVariant(self.autoHideAction.isChecked()))
        config.get_settings().setValue(config.KEY_DEMOS_DIR, QtCore.QVariant(self.demosDir))
                              

    def loadLayout(self):
        '''Load the window layout.'''
        geo_data = config.get_settings().value(config.KEY_WINDOW_GEOMETRY).toByteArray()
        layout_data = config.get_settings().value(config.KEY_WINDOW_LAYOUT).toByteArray()
        self.restoreGeometry(geo_data)
        self.restoreState(layout_data)
        # The checked state of the menu items do not remain stored
        # properly. Looks like something dependent on the sequence of
        # object creation. So after every restart the GLClient will be
        # visible while TreePanel depends on what state ot was in when
        # the application was closed last time. The following code
        # fixes that issue.
        if self.mooseTreePanel.isHidden():
            self.mooseTreeAction.setChecked(False)
        else:
            self.mooseTreeAction.setChecked(True)

        # if self.glClientDock.isHidden():
        #     # print 'Glclient is hidden'
        #     self.glClientAction.setChecked(False)
        # else:
        #     # print 'Glclient is visible'
        #     self.glClientAction.setChecked(True)
        if self.mooseClassesPanel.isHidden():
            self.mooseClassesAction.setChecked(False)
        else:
            self.mooseClassesAction.setChecked(True)
        if self.commandLineDock.isHidden():
            self.mooseShellAction.setChecked(False)
        else:
            self.mooseShellAction.setChecked(True)
        
    
    def createMooseClassesPanel(self):
        config.LOGGER.debug('createMooseClassesPanel - start')
        self.mooseClassesPanel = QtGui.QDockWidget(self.tr('Classes'), self)
        self.mooseClassesPanel.setObjectName(self.tr('MooseClassPanel'))
	self.mooseClassesPanel.hide()#add_chait
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.mooseClassesPanel)
	self.mooseClassesWidget = makeClassList(self.mooseClassesPanel)
	self.mooseClassesPanel.setWidget(self.mooseClassesWidget)
        config.LOGGER.debug('createMooseClassesPanel - end')

    def createMooseTreePanel(self):
        config.LOGGER.debug('createMooseTreePanel - start')
	self.mooseTreePanel = QtGui.QDockWidget(self.tr('Element Tree'), self)
        self.mooseTreePanel.setObjectName(self.tr('MooseClassPanel'))
	#~ self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.mooseTreePanel)
        self.mooseTreePanel.hide()#add_chait
	self.modelTreeWidget = MooseTreeWidget(self.mooseTreePanel)
        self.modelTreeWidget.setMooseHandler(self.mooseHandler)
        # self.modelTreeWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        # self.connect(self.modelTreeWidget, QtCore.SIGNAL('customContextMenuRequested ( const QPoint&)'), self.createConnectionMenu)
	self.mooseTreePanel.setWidget(self.modelTreeWidget)
        config.LOGGER.debug('createMooseTreePanel - end')
        self.ForceTabbedDocks
        self.makeObjectFieldEditor(self.modelTreeWidget.currentItem().getMooseObject())
        
    def createGLClientDock(self):
        config.LOGGER.debug('createGLClientDock - start')
        self.glClientWidget = GLClientGUI(self)
        config.LOGGER.debug('createGLClientDock - 1')
        self.glClientDock = QtGui.QDockWidget('GL Client', self)
        config.LOGGER.debug('createGLClientDock - 2')
        self.glClientDock.setObjectName(self.tr('GLClient'))
        config.LOGGER.debug('createGLClientDock - 3')
        self.glClientDock.setWidget(self.glClientWidget)
        config.LOGGER.debug('createGLClientDock - 4')
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.glClientDock)
        config.LOGGER.debug('createGLClientDock - end')

    def createControlDock(self):
        config.LOGGER.debug('Making control panel')
        self.controlDock = QtGui.QDockWidget(self.tr('Simulation Control'), self)
        self.controlDock.setObjectName(self.tr('Control Dock'))

        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.controlDock)
        self.controlPanel = QtGui.QFrame(self)
        self.controlPanel.setFrameStyle(QtGui.QFrame.StyledPanel | QtGui.QFrame.Plain)
        layout = QtGui.QGridLayout()
        self.runtimeLabel = QtGui.QLabel(self.tr('Simulation Run Time (second):'), self.controlPanel)
        self.runtimeLabel.setWordWrap(True)
        self.runtimeText = QtGui.QLineEdit('%1.3e' % (MooseHandler.runtime), self.controlPanel)
        self.updateTimeLabel = QtGui.QLabel(self.tr('Update interval for plots & viz (second):'), self.controlPanel)
        self.updateTimeLabel.setWordWrap(True)
        self.updateTimeText = QtGui.QLineEdit('%1.3e' % (MooseHandler.plotupdate_dt), self.controlPanel)
        # self.resetButton = QtGui.QPushButton(self.tr('Reset'), self.controlPanel)
        self.runButton = QtGui.QPushButton(self.tr('Run'), self.controlPanel)
        self.continueButton = QtGui.QPushButton(self.tr('Continue'), self.controlPanel)
        self.simdtLabel = QtGui.QLabel(self.tr('Simulation timestep (second):'), self.controlPanel)
        self.simdtLabel.setWordWrap(True)
        self.plotdtLabel = QtGui.QLabel(self.tr('Plotting timestep (second):'), self.controlPanel)
        self.plotdtLabel.setWordWrap(True)
        #self.gldtLabel = QtGui.QLabel(self.tr('3D visualization timestep (second):'), self.controlPanel)
        self.simdtText = QtGui.QLineEdit('%1.3e' % (MooseHandler.simdt), self.controlPanel)
        self.plotdtText = QtGui.QLineEdit('%1.3e' % (MooseHandler.plotdt), self)
        #self.gldtText = QtGui.QLineEdit('%1.3e' % (MooseHandler.gldt), self.controlPanel)
        # self.overlayCheckBox = QtGui.QCheckBox(self.tr('Overlay plots'), self.controlPanel)
        
        self.connect(self.runButton, QtCore.SIGNAL('clicked()'), self.resetAndRunSlot)
        self.connect(self.continueButton, QtCore.SIGNAL('clicked()'), self._runSlot)
        layout.addWidget(self.simdtLabel, 0,0)
        layout.addWidget(self.simdtText, 0, 1)
        layout.addWidget(self.plotdtLabel, 1, 0)
        layout.addWidget(self.plotdtText, 1, 1)
        #layout.addWidget(self.gldtLabel, 2, 0)
        #layout.addWidget(self.gldtText, 2, 1)
        # layout.addWidget(self.overlayCheckBox, 3, 0)
        layout.addWidget(self.runtimeLabel, 4, 0)
        layout.addWidget(self.runtimeText, 4, 1)
        layout.addWidget(self.updateTimeLabel, 5, 0)
        layout.addWidget(self.updateTimeText, 5,1)
        layout.addWidget(self.runButton, 6, 0)
        layout.addWidget(self.continueButton, 6, 1)
        self.controlPanel.setLayout(layout)
        self.controlDock.setWidget(self.controlPanel)
        self.controlDock.hide()#add_chait

    def addPlotWindow(self):
        title = self.tr('Plot %d' % (len(self.plots)))
        plotWindow = MoosePlotWindow()
        plotWindow.setWindowTitle(title)
        plotWindow.setObjectName(title)
        plot = MoosePlot(plotWindow)
        plot.mooseHandler = self.mooseHandler
        plot.setObjectName(title)
        plotWindow.setWidget(plot)
        self.plots.append(plot)
        if len(self.plots)>1:
            self.objFieldEditModel.plotNames.append(plot.objectName())
        self.centralPanel.addSubWindow(plotWindow)
        plotWindow.setAcceptDrops(True)
        self.connect(plotWindow, QtCore.SIGNAL('subWindowClosed()'), self.decrementSubWindowCount)
        # self.plotWindows.append(plotWindow)
        self.centralPanel.setActiveSubWindow(plotWindow)
        plotWindow.show()
        self._visiblePlotWindowCount += 1
        # if hasattr(self, 'cascadePlotWindowsAction') and self.cascadePlotWindowsAction.isChecked():
        #     self.centralPanel.cascadeSubWindows()
        # else:
        #     self.centralPanel.tileSubWindows()
        self.currentPlotWindow = plotWindow
        return plotWindow
        
    def addGLWindow(self):   #add_chait
       
        self.newDia = QtGui.QDialog(self)	
        self.vizDialogue = Ui_Dialog()
        self.vizDialogue.setupUi(self.newDia)
        self.newDia.show()
        
        self.connect(self.vizDialogue.resetButton, QtCore.SIGNAL('clicked()'), self.resetVizDialogSettings)
        self.connect(self.vizDialogue.acceptButton, QtCore.SIGNAL('clicked()'),  self.vizSettings)
        self.connect(self.vizDialogue.mtree, QtCore.SIGNAL('itemDoubleClicked(QTreeWidgetItem *, int)'),self.updateVizCellList)
        self.connect(self.vizDialogue.addCellButton, QtCore.SIGNAL('clicked()'),self.addCellToVizList)
        self.connect(self.vizDialogue.removeCellButton, QtCore.SIGNAL('clicked()'),self.removeCellFromVizList)
        self.connect(self.vizDialogue.allCellsButton, QtCore.SIGNAL('clicked()'),self.addAllCellsToVizList)
        self.connect(self.vizDialogue.styleComboBox,QtCore.SIGNAL('currentIndexChanged(int)'),self.styleComboChange)
    
    def styleComboChange(self,a):	#add_chait
    	if a==3:
    		self.vizDialogue.specificCompartmentName.setEnabled(True)
    		self.vizDialogue.label_7.setEnabled(True)
    		self.vizDialogue.label_11.setEnabled(True)
    		self.vizDialogue.label_12.setEnabled(True)
    		self.vizDialogue.label_13.setEnabled(True)
    		self.vizDialogue.label_14.setEnabled(True)
    		self.vizDialogue.variable_2.setEnabled(True)
    		self.vizDialogue.moosepath_2.setEnabled(True)
    		self.vizDialogue.vizMaxVal_2.setEnabled(True)
    		self.vizDialogue.vizMinVal_2.setEnabled(True)
    	elif a==0:
    		self.vizDialogue.specificCompartmentName.setEnabled(True)
    		self.vizDialogue.label_7.setEnabled(True)
    		self.vizDialogue.label_11.setEnabled(False)
    		self.vizDialogue.label_12.setEnabled(False)
    		self.vizDialogue.label_13.setEnabled(False)
    		self.vizDialogue.label_14.setEnabled(False)
    		self.vizDialogue.variable_2.setEnabled(False)
    		self.vizDialogue.moosepath_2.setEnabled(False)
    		self.vizDialogue.vizMaxVal_2.setEnabled(False)
    		self.vizDialogue.vizMinVal_2.setEnabled(False)
    	else:
    		self.vizDialogue.specificCompartmentName.setEnabled(False)
        	self.vizDialogue.label_7.setEnabled(False)
        	self.vizDialogue.label_11.setEnabled(False)
    		self.vizDialogue.label_12.setEnabled(False)
    		self.vizDialogue.label_13.setEnabled(False)
    		self.vizDialogue.label_14.setEnabled(False)
    		self.vizDialogue.variable_2.setEnabled(False)
    		self.vizDialogue.moosepath_2.setEnabled(False)
    		self.vizDialogue.vizMaxVal_2.setEnabled(False)
    		self.vizDialogue.vizMinVal_2.setEnabled(False)
        	
    def addAllCellsToVizList(self):		#add_chait
        an=moose.Neutral('/')						#moose root children
	all_ch=an.childList 	
						#all children under root, of cell type
	ch = self.get_childrenOfField(all_ch,'Cell')
	for i in range(0,len(ch),1):
		if self.vizDialogue.vizCells.count() == 0:
        		self.vizDialogue.vizCells.addItem(moose.Cell(ch[i]).path)
        	else:
	    		for index in xrange(self.vizDialogue.vizCells.count()):
	    			if str(self.vizDialogue.vizCells.item(index).text())!= moose.Cell(ch[i]).path:
	    				self.vizDialogue.vizCells.addItem(moose.Cell(ch[i]).path)
	    
	nh = self.get_childrenOfField(all_ch,'Neutral')			#all cells under all other neutral elements.	
	for j in range(0,len(nh),1):
	    an=moose.Neutral(nh[j])					#this neutral element
	    all_ch=an.childList 					#all children under this neutral element
	    ch = self.get_childrenOfField(all_ch,'Cell')
	    for i in range(0,len(ch),1):
	    	if self.vizDialogue.vizCells.count() == 0:
        		self.vizDialogue.vizCells.addItem(moose.Cell(ch[i]).path)
        	else:
	    		for index in xrange(self.vizDialogue.vizCells.count()):
	    			if str(self.vizDialogue.vizCells.item(index).text())!= moose.Cell(ch[i]).path:
	    				self.vizDialogue.vizCells.addItem(moose.Cell(ch[i]).path)
	    	
    def get_childrenOfField(self,all_ch,field):	##add_chait
        ch=[]
        for i in range(0,len(all_ch)):	
	    if(mc.className(all_ch[i])==field):
	        ch.append(all_ch[i])
        return tuple(ch)  
        
    def removeCellFromVizList(self):		#add_chait
    	self.vizDialogue.vizCells.takeItem(self.vizDialogue.vizCells.currentRow())
        
    def addCellToVizList(self):			#add_chait
    	self.updateVizCellList(self.vizDialogue.mtree.currentItem(),1)

    def updateVizCellList(self, item, column):	#add_chait
        if item.mooseObj_.className == 'Cell':
        	if self.vizDialogue.vizCells.count() == 0:
        		self.vizDialogue.vizCells.addItem(item.mooseObj_.path)
        	else:
        		for index in xrange(self.vizDialogue.vizCells.count()):
     				if str(self.vizDialogue.vizCells.item(index).text())!= item.mooseObj_.path:
        				self.vizDialogue.vizCells.addItem(item.mooseObj_.path)
		
    
    def resetVizDialogSettings(self):	#add_chait
	self.vizDialogue.variable.setText("Vm")
	self.vizDialogue.moosepath.setText("")
        self.vizDialogue.vizMinVal.setText("-0.1")
        self.vizDialogue.vizMaxVal.setText("0.07")
        self.vizDialogue.variable_2.setText("")
	self.vizDialogue.moosepath_2.setText("")
        self.vizDialogue.vizMinVal_2.setText("")
        self.vizDialogue.vizMaxVal_2.setText("")
        self.vizDialogue.colorMapComboBox.setCurrentIndex(0)
        self.vizDialogue.styleComboBox.setCurrentIndex(2)
    
    def vizSettings(self):   #add_chait
    	
    	title = self.tr('GL %d' % (len(self.vizs)))
        vizWindow = newGLSubWindow()
        vizWindow.setWindowTitle(title)
        vizWindow.setObjectName(title)
        
        self.centralVizPanel.addSubWindow(vizWindow)
        viz = updatepaintGL(parent=vizWindow)
        viz.setObjectName(title)
        vizWindow.setWidget(viz)
        
        viz.viz=1	#turn on visualization mode
	vizStyle = self.vizDialogue.styleComboBox.currentIndex()
	
	if self.vizDialogue.specificCompartmentName.text()!='':					#if no compartment name selected, default is soma
       		viz.specificCompartmentName = str(self.vizDialogue.specificCompartmentName.text())	#else pick name from user ip text box
	
	if vizStyle==3:    									#grid view case
		numberOfCellsDrawn = 0								#as yet drawn = 0, used as a counter
		sideSquare = self.nearestSquare(self.vizDialogue.vizCells.count())		#get the side of the square
		for yAxis in range(sideSquare):							#Yaxis - columns
			for xAxis in range(sideSquare):						#Xaxis - fill rows first
				if numberOfCellsDrawn < self.vizDialogue.vizCells.count(): 	#finished drawing all cells?
					viz.drawNewCell(cellName=str(self.vizDialogue.vizCells.item(numberOfCellsDrawn).text()),cellCentre=[xAxis*0.5,yAxis*0.5,0.0],style = vizStyle)
					numberOfCellsDrawn += 1					#increase number of cells drawn
			
       		if self.vizDialogue.variable_2.text()!='':					#field2 to represented as radius of the compartment
       			currentVizSetting_2 = [float(str(self.vizDialogue.vizMinVal_2.text())),float(str(self.vizDialogue.vizMaxVal_2.text())),str(self.vizDialogue.moosepath_2.text()),str(self.vizDialogue.variable_2.text())]
       			viz.setColorMap_2(*currentVizSetting_2[:4])				#set the colormap equivalent of radius
			viz.gridRadiusViz=1							#viz using both radius and color - 2 fields
		else:										
			viz.gridRadiusViz=0							#no 2nd field selected, just viz using colors - not radius

	else:
    		for index in xrange(self.vizDialogue.vizCells.count()):				#non-grid view case
        		viz.drawNewCell(cellName=str(self.vizDialogue.vizCells.item(index).text()),style = vizStyle)
        
        currentVizSetting = [float(str(self.vizDialogue.vizMinVal.text())),float(str(self.vizDialogue.vizMaxVal.text())),str(self.vizDialogue.moosepath.text()),str(self.vizDialogue.variable.text()),os.path.join(str(self.settings.value(config.KEY_GL_COLORMAP).toString()),str(self.vizDialogue.colorMapComboBox.itemText(self.vizDialogue.colorMapComboBox.currentIndex())))]						#color map inputs from user
    	#print currentVizSetting
        
        viz.setColorMap(*currentVizSetting[:5])							#assign regular colormap
        
        QtCore.QObject.connect(viz,QtCore.SIGNAL("compartmentSelected(QString)"),self.pickCompartment)
        #viz.translate([0.0,0.0,-6.0])
        self.vizs.append(viz)
        
        self.newDia.hide() 	#pressed OK button, so close the dialog
        self.connect(vizWindow, QtCore.SIGNAL('subWindowClosed()'), self.decrementSubWindowCount)
        self.centralVizPanel.setActiveSubWindow(vizWindow)
        vizWindow.show()
        #print vizWindow.width()
        #print vizWindow.height()
        #vizWindow.setFixed
        vizWindow.showMaximized()
        self._visiblePlotWindowCount += 1
        self.currentPlotWindow = vizWindow
        return vizWindow
   
    def pickCompartment(self,path):	#path is a QString type moosepath
        SelectedChild = self.modelTreeWidget.pathToTreeChild(path)
    	self.modelTreeWidget.setCurrentItem(SelectedChild)				#select the corresponding moosetree
	self.makeObjectFieldEditor(SelectedChild.getMooseObject())		#update the corresponding property

    def nearestSquare(self, n):	#add_chait
    	i = 1
	while i * i < n:
		i += 1
	return i

    def setPlotWindowsVisible(self, on=True):
        """Toggle visibility of plot windows.

        """
        if on:
            for window in self.centralPanel.subWindowList():
                window.show()
            self._visiblePlotWindowCount = len(self.centralPanel.subWindowList())
        else:
            print 'Closing all subwindows'
            self.centralPanel.closeAllSubWindows()
            self._visiblePlotWindowCount = 0

    def switchTabbedView(self, checked):
        if checked:
            self.centralPanel.setViewMode(self.centralPanel.TabbedView)
        else:
            self.centralPanel.setViewMode(self.centralPanel.SubWindowView)
            
        

    def addLayoutWindow(self):
    	centralWindowsize =  self.centralVizPanel.size()
        self.sceneLayout = layout.LayoutWidget(centralWindowsize)
	self.connect(self.sceneLayout, QtCore.SIGNAL("itemDoubleClicked(PyQt_PyObject)"), self.makeObjectFieldEditor)
        self.centralVizPanel.addSubWindow(self.sceneLayout)
	self.centralVizPanel.tileSubWindows()
        self.sceneLayout.show()
    

    def decrementSubWindowCount(self):
        if self._visiblePlotWindowCount > 0:
            self._visiblePlotWindowCount -= 1
        if self._visiblePlotWindowCount == 0:
            self.togglePlotWindowsAction.setChecked(False)

    def popupLoadModelDialog(self):
        fileDialog = QtGui.QFileDialog(self)        
        fileDialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        ffilter = ''
        for key in sorted(self.mooseHandler.fileExtensionMap.keys()):
            ffilter = ffilter + key + ';;'
        ffilter = ffilter[:-2]
        fileDialog.setFilter(self.tr(ffilter))
        # The following version gymnastic is because QFileDialog.selectNameFilter() was introduced in Qt 4.4
        if (config.QT_MAJOR_VERSION > 4) or ((config.QT_MAJOR_VERSION == 4) and (config.QT_MINOR_VERSION >= 4)):
            for key, value in self.mooseHandler.fileExtensionMap.items():
                if value == MooseHandler.type_genesis:
                    fileDialog.selectNameFilter(key)
                    break
        targetPanel = QtGui.QFrame(fileDialog)
        targetPanel.setLayout(QtGui.QVBoxLayout())
        targetTree = MooseTreeWidget(fileDialog)
        currentPath = self.mooseHandler._current_element.path
        for item in targetTree.itemList:
            if item.getMooseObject().path == currentPath:
                targetTree.setCurrentItem(item)
        targetLabel = QtGui.QLabel('Target Element')
        targetText = QtGui.QLineEdit(fileDialog)
        targetText.setText(currentPath)
        targetPanel.layout().addWidget(targetLabel)
        targetPanel.layout().addWidget(targetText)
        layout = fileDialog.layout()
        layout.addWidget(targetTree)
        layout.addWidget(targetPanel)
        self.connect(targetTree, QtCore.SIGNAL('itemClicked(QTreeWidgetItem *, int)'), lambda item, column: targetText.setText(item.getMooseObject().path))
	if fileDialog.exec_():
	    fileNames = fileDialog.selectedFiles()
	    fileFilter = fileDialog.selectedFilter()
	    fileType = self.mooseHandler.fileExtensionMap[str(fileFilter)]
	    directory = fileDialog.directory() # Potential bug: if user types the whole file path, does it work? - no but gives error message
	    for fileName in fileNames: 
                modeltype  = self.mooseHandler.loadModel(str(fileName), str(fileType), str(targetText.text()))
		if modeltype == MooseHandler.type_kkit:
		    self.addLayoutWindow()
                print 'Loaded model',  fileName, 'of type', modeltype
            self.populateKKitPlots()
            self.populateDataPlots()
            self.updateDefaultTimes(modeltype)
            self.modelTreeWidget.recreateTree()
        
        self.checkModelType()
        
    #add_chait
    def checkModelType(self):
        an=moose.Neutral('/')						#moose root children
	all_ch=an.childList 	
        ch = self.get_childrenOfField(all_ch,'Cell')
        if ch :#if has cell type child elements.
            #loaded model is a cell model, plot the cells in the 
            if len(ch)==1:
                #only the single cell models to be visualized
                title = self.tr('GL %d' % (len(self.vizs)))
                vizWindow = newGLSubWindow()
                vizWindow.setWindowTitle(title)
                vizWindow.setObjectName(title)
                self.centralVizPanel.addSubWindow(vizWindow)
                viz = updatepaintGL(parent=vizWindow)
                viz.setObjectName(title)
                vizWindow.setWidget(viz)
        
                viz.viz=1	#turn on visualization mode
                viz.drawNewCell(cellName=moose.Cell(ch[0]).path,style = 2)
                viz.setColorMap(cMap=os.path.join(str(self.settings.value(config.KEY_GL_COLORMAP).toString()),'jet'))
                QtCore.QObject.connect(viz,QtCore.SIGNAL("compartmentSelected(QString)"),self.pickCompartment)
                self.vizs.append(viz)
                vizWindow.show()
                vizWindow.showMaximized()
                self._visiblePlotWindowCount += 1
                self.currentPlotWindow = vizWindow
                return vizWindow
   

    def resetSettings(self):
        self.settingsReset = True
        config.get_settings().clear()

    def setCurrentElement(self, item, column):
        """Set the current object of the mooseHandler"""
        current_element = item.getMooseObject()
        self.mooseHandler._current_element = current_element
        
    def _resetSlot(self):
        """Get the dt-s from the UI and call the reset method in
        MooseHandler
        """
        try:
            simdt = float(str(self.simdtText.text()))
        except ValueError, simdt_err:
            print 'Error setting Simulation timestep:', simdt_err
            simdt = MooseHandler.simdt
            self.simdtText.setText('%1.3e' % (MooseHandler.simdt))
        try:
            plotdt = float(str(self.plotdtText.text()))
        except ValueError, plotdt_err:
            print 'Error setting Plotting timestep:', plotdt_err
            plotdt = MooseHandler.plotdt
            self.plotdtText.setText('%1.3e' % (MooseHandler.plotdt))
        #try:            
        #    gldt = float(str(self.gldtText.text()))
        #except ValueError, gldt_err:
        #    print 'Error setting 3D visualization time step:', gldt_err
        #    gldt = MooseHandler.gldt
        #    self.gldtText.setText('%1.3e' % (MooseHandler.gldt))
        try:
            updateInterval = float(str(self.updateTimeText.text()))
            if updateInterval < 0.0:
                updateInterval = MooseHandler.plotupdate_dt
        except ValueError:
            updateInterval = MooseHandler.plotupdate_dt
            self.updateTimeText.setText(str(updateInterval))
        #self.mooseHandler.doReset(simdt, plotdt, gldt,updateInterval)
        self.mooseHandler.doReset(simdt, plotdt, updateInterval)
        for table, plot in self.tablePlotMap.items():
            print 'Clearing plot', table.name
            table.clear()
            # plot.setOverlay(self.overlayCheckBox.isChecked())
            plot.reset()
                    

    def _runSlot(self):
        """Run the simulation.

        """
        if self.autoHideAction.isChecked():
            if self.commandLineDock.isVisible():
                self.commandLineDock.setVisible(False)
            if self.mooseClassesPanel.isVisible():
                self.mooseClassesPanel.setVisible(False)
            # if self.glClientDock.isVisible():
            #     self.glClientDock.setVisible(False)
            if hasattr(self, 'objFieldEditPanel') and self.objFieldEditPanel.isVisible():
                self.objFieldEditPanel.setVisible(False)
            self.showRightBottomDocksAction.setChecked(False)
        try:
            runtime = float(str(self.runtimeText.text()))
        except ValueError:
            runtime = MooseHandler.runtime
            self.runtimeText.setText(str(runtime))
        self.updatePlots(runtime)
        self.mooseHandler.doRun(runtime)

    #add_chait
    def resetAndRunSlot1(self): #horrible way of doing it. because of simulation toolbar.
        self._resetSlot()
        self._runSlot1()

    #add_chait    
    def _runSlot1(self):#bad way of doing it. but works just fine. (this is because of the simulation toolbar)
        """Run the simulation.

        """
        if self.autoHideAction.isChecked():
            if self.commandLineDock.isVisible():
                self.commandLineDock.setVisible(False)
            if self.mooseClassesPanel.isVisible():
                self.mooseClassesPanel.setVisible(False)
            # if self.glClientDock.isVisible():
            #     self.glClientDock.setVisible(False)
            if hasattr(self, 'objFieldEditPanel') and self.objFieldEditPanel.isVisible():
                self.objFieldEditPanel.setVisible(False)
            self.showRightBottomDocksAction.setChecked(False)
        try:
            runtime = float(str(self.runTimeEditToolbar.text()))
        except ValueError:
            runtime = MooseHandler.runtime
            self.runtimeText.setText(str(runtime))
        self.updatePlots(runtime)
        self.mooseHandler.doRun(runtime)

              
    def resetAndRunSlot(self):
        self._resetSlot()
        self._runSlot()

    def changeFieldPlotWidget(self, full_field_path, plotname):
        """Remove the plot for the specified field from the current
        plot window and set it to the plotwindow with given name."""
        fieldpath = str(full_field_path)
        for plot in self.plots:
            if plotname == plot.objectName():
                table = self.mooseHandler.addFieldTable(fieldpath)
                plot.addTable(table)
                try:
                    oldplot = self.tablePlotMap[table]
                    oldplot.removeTable(table)
                except KeyError:
                    pass
                self.tablePlotMap[table] = plot
                plot.replot()
                

    def updatePlots(self, currentTime):
        for plot in self.plots:
            plot.updatePlot(currentTime)
        self.updateVizs()	#add_chait  
        self.updateCurrentTime(currentTime)

    def updateCurrentTime(self,currentTime): #add_chait
        self.currentTimeEditToolbar.setText(str(currentTime))

    def updateVizs(self):	#add_chait
    	for viz in self.vizs:
       		viz.updateViz()

    def changeShellMode(self, action):
        if action == self.pythonModeAction:
            self.shellWidget.setMode(MooseGlobals.CMD_MODE_PYMOOSE)
        elif action == self.genesisModeAction:
            self.shellWidget.setMode(MooseGlobals.CMD_MODE_GENESIS)
        else:
            config.LOGGER.error('Unknown action: %s' % (action.text()))

    def setConnSrc(self):
        sender = QtCore.QObject.sender()
        path = str(sender.data().toString())
        self.mooseHandler.setConnSrc(path)
        self.mooseHandler.doConnect()

    def setConnDest(self):
        sender = QtCore.QObject.sender()
        path = str(sender.data().toString())
        self.mooseHandler.setConnDest(path)
        self.mooseHandler.doConnect()

    def configurePlots(self):
        """Interactively allow the user to configure everything about
        the plots."""
        self.plotConfig.setVisible(True)
        ret = self.plotConfig.exec_()
        # print ret, QtGui.QDialog.Accepted
        if ret == QtGui.QDialog.Accepted:
            pen = self.plotConfig.getPen()
            symbol = self.plotConfig.getSymbol()
            style = self.plotConfig.getStyle()
            attribute = self.plotConfig.getAttribute()
            activePlot = self.currentPlotWindow            
            plotName = activePlot.objectName() #windowTitle() # The window title is the plot name
            # print 'configurePlots', plotName 
            for plot in self.plots:
                # print 'plot.objectName:', plot.objectName(), 'plotName:', plotName
                if plot.objectName() == plotName:
                    plot.reconfigureSelectedCurves(pen, symbol, style, attribute)
                    break

    def togglePlotVisibility(self, hide):
        print 'Currently selected to hide?', hide
        activePlot = self.currentPlotWindow            
        plotName = activePlot.objectName()#windowTitle() # The window title is the plot name
        for plot in self.plots:
            print plot.objectName(), '#', plotName
            if plot.objectName() == plotName:
                plot.showSelectedCurves(not hide)
                break

    def showAllPlots(self):
        for plot in self.plots:
            plot.showAllCurves()

    def createConnection(self):
        if (self._srcElement is None) or (self._destElement is None):
            return
        src = self._srcElement.path + '/' + str(self.sourceFieldComboBox.currentText())
        dest = self._destElement.path + '/' + str(self.destFieldComboBox.currentText())
        self.mooseHandler.setConnSrc(src)
        self.mooseHandler.setConnDest(dest)
        ret = self.mooseHandler.doConnect()
        # print 'Connecting %s to %s: %s' % (src, dest, 'success' if ret else 'failed')        
        self._srcElement = None
        self._destElement = None
        


    def setCurrentPlotWindow(self, subWindow):
        if subWindow:
            self.currentPlotWindow = subWindow.widget()

    def startGLWizard(self):
        self.glWizard = MooseGLWizard(self, self.mooseHandler)
        self.glWizard.setVisible(True)

    def populateKKitPlots(self):
        graphs = self.mooseHandler.getKKitGraphs()
        for graph in graphs:
            self.plots[0].addTable(graph)
            config.LOGGER.info('Adding plot ' + graph.path)
        self.plots[0].replot()
        moregraphs =self.mooseHandler.getKKitMoreGraphs()
        if len(moregraphs) > 0:
            if len(self.plots) == 1:
                self.addPlotWindow()
            for graph in moregraphs:
                self.plots[1].addTable(graph)
            self.plots[1].replot()

    def populateDataPlots(self):
        """Create plots for all Table objects in /data element"""
        tables = self.mooseHandler.getDataTables()
        for table in tables:
            self.plots[0].addTable(table)
            config.LOGGER.info('Added plot ' + table.path)
        self.plots[0].replot()

    def startFirstTimeWizard(self):
	print 'Starting first time wizard'
        firstTimeWizard = FirstTimeWizard(self)
        self.connect(firstTimeWizard, QtCore.SIGNAL('accepted()'), self.updatePaths)
        firstTimeWizard.show()

    def doQuit(self):
        self.mooseHandler.stopGL()
        QtGui.qApp.closeAllWindows()

    def selectConnSource(self, item, column):
        self._srcElement = item.getMooseObject()
        self.sourceObjText.setText(self._srcElement.path)
        self.sourceFieldComboBox.addItems(self.mooseHandler.getSrcFields(self._srcElement))

    def selectConnDest(self, item, column):
        self._destElement = item.getMooseObject()
        self.destObjText.setText(self._destElement.path)
        self.destFieldComboBox.addItems(self.mooseHandler.getDestFields(self._destElement))

    def updatePaths(self):
        self.demosDir = str(config.get_settings().value(config.KEY_DEMOS_DIR).toString())
	config.LOGGER.info('Demos directory: %s' % self.demosDir)
        

    def renameObjFieldEditPanel(self, mooseObject):
        self.objFieldEditPanel.setWindowTitle(mooseObject.name)

    def browseDocumentation(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(QtCore.QString(config.MOOSE_DOC_URL)))

    def openBugsPage(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(QtCore.QString(config.MOOSE_REPORT_BUG_URL)))

    def updateDefaultTimes(self, modeltype):
        self.simdtText.setText(QtCore.QString('%1.3e' % (MooseHandler.DEFAULT_SIMDT)))
        if (modeltype == MooseHandler.type_kkit) or (modeltype == MooseHandler.type_sbml):
            self.simdtText.setText(QtCore.QString('%1.3e' % (MooseHandler.DEFAULT_SIMDT_KKIT)))
            self.plotdtText.setText(QtCore.QString('%1.3e' % (MooseHandler.DEFAULT_PLOTDT_KKIT)))
            #self.gldtText.setText(QtCore.QString('%1.3e' % (MooseHandler.DEFAULT_GLDT_KKIT)))
            self.runtimeText.setText(QtCore.QString('%1.3e' % (MooseHandler.DEFAULT_RUNTIME_KKIT)))
            self.updateTimeText.setText(QtCore.QString('%1.3e' % (MooseHandler.DEFAULT_PLOTUPDATE_DT_KKIT)))

            

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    icon = QtGui.QIcon('moose_icon.png')
    app.setWindowIcon(icon)

    QtCore.QObject.connect(app, QtCore.SIGNAL('lastWindowClosed()'), app, QtCore.SLOT('quit()'))
    mainWin = MainWindow()
    if not config.get_settings().contains(config.KEY_FIRSTTIME):
        firstTimeWizard = FirstTimeWizard()
        firstTimeWizard.setModal(QtCore.Qt.ApplicationModal)
	firstTimeWizard.connect(firstTimeWizard, QtCore.SIGNAL('accepted()'), mainWin.updatePaths)
        firstTimeWizard.show()
    mainWin.show()
    app.exec_()
	
