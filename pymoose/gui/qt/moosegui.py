#!/usr/bin/env python
# moosegui.py --- 
# 
# Filename: moosegui.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Wed Jan 20 15:24:05 2010 (+0530)
# Version: 
# Last-Updated: Tue Jul 13 10:56:08 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 2254
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

import os
import sys
import code
import subprocess
from datetime import date
from collections import defaultdict

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
from objectedit import ObjectFieldsModel, ObjectEditDelegate
from moosetree import *
from mooseclasses import *
from mooseglobals import MooseGlobals
from mooseshell import MooseShell
from moosehandler import MooseHandler
from mooseplot import MoosePlot
from plotconfig import PlotConfig
from glwizard import MooseGLWizard
from firsttime import FirstTimeWizard


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


# class ConnStruct:
#     def __init__(self, srcElm=None, srcField=None, destElm=None, destField=None):
#         self.srcElm = srcElm
#         self.destElm = destElm
#         self.srcField = srcField
#         self.destField = destField


class MainWindow(QtGui.QMainWindow):
    default_plot_count = 1
    def __init__(self, interpreter=None, parent=None):
	QtGui.QMainWindow.__init__(self, parent)
        self.mooseHandler = MooseHandler()
        self.connect(self.mooseHandler, QtCore.SIGNAL('updatePlots(float)'), self.updatePlots)
        self.settings = config.get_settings()        
        self.demosDir = str(self.settings.value(config.KEY_DEMOS_DIR).toString())
        if not self.demosDir:
            self.demosDir = str(QtGui.QFileDialog.getExistingDirectory(self, 'Please select pymoose demos directory'))
        self.resize(800, 600)
        self.setDockOptions(self.AllowNestedDocks | self.AllowTabbedDocks | self.ForceTabbedDocks | self.AnimatedDocks)        
        # The following are for holding transient selections from
        # connection dialog
        self._srcElement = None
        self._destElement = None
        self._srcField = None
        self._destField = None
        # This is the element tree of MOOSE
        self.createMooseTreePanel()
        # List of classes - one can double click on any class to
        # create an instance under the currently selected element in
        # mooseTreePanel
        self.createMooseClassesPanel()
        # Create a widget to configure the glclient
        self.createGLClientDock()
        self.createControlDock()
        # Connect the double-click event on modelTreeWidget items to
        # creation of the object editor.
        # TODO - will create a right-click menu
        self.connect(self.modelTreeWidget, 
                     QtCore.SIGNAL('itemDoubleClicked(QTreeWidgetItem *, int)'),
                     self.makeObjectFieldEditor)
        self.connect(self.modelTreeWidget, 
                     QtCore.SIGNAL('itemClicked(QTreeWidgetItem *, int)'),
                     self.setCurrentElement)
        # We use the objFieldEditorMap to store reference to cache the
        # model objects for moose objects.
        self.objFieldEditorMap = {}
        self.makeShellDock(interpreter)
        # By default, we show information about MOOSE in the central widget
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.centralPanel = QtGui.QMdiArea(self)
        # plots is a list of available MoosePlot widgets.
        self.plots = []
        # tablePlotMap is a maps all currently available tables to the
        # plot widgets they belong to.
        self.tablePlotMap = {}
        self.currentPlotWindow = None
        # Start with a default number of plot widgets
        for ii in range(MainWindow.default_plot_count):
            self.addPlotWindow()
        self.plotConfig = PlotConfig(self)
        self.plotConfig.setVisible(False)        
        self.setCentralWidget(self.centralPanel)
        self.centralPanel.tileSubWindows()
        self.centralPanel.subWindowActivated.connect(self.setCurrentPlotWindow)
        # We connect the double-click event on the class-list to
        # insertion of moose object in model tree.
        for listWidget in self.mooseClassesWidget.getClassListWidget():
            self.connect(listWidget, 
                         QtCore.SIGNAL('itemDoubleClicked(QListWidgetItem*)'), 
                         self.insertMooseObjectSlot)
        self.connect(QtGui.qApp, QtCore.SIGNAL('lastWindowClosed()'), self.saveLayout)
        self.createActions()
        self.makeMenu()
        # Loading of layout should happen only after all dockwidgets
        # have been created
        self.loadLayout()

    def makeConnectionPopup(self):
        """Create a dialog to connect moose objects via messages."""
        self.connectionDialog = QtGui.QDialog(self)
        self.connectionDialog.setWindowTitle(self.tr('Create Connection'))
        self.connectionDialog.setModal(False)

        self.sourceTree = MooseTreeWidget(self.connectionDialog)
        self.destTree = MooseTreeWidget(self.connectionDialog)
        self.sourceTree.itemClicked.connect(self.selectConnSource)
        self.destTree.itemClicked.connect(self.selectConnDest)
        sourceFieldLabel = QtGui.QLabel(self.tr('Source Field'), self.connectionDialog)
        self.sourceFieldComboBox = QtGui.QComboBox(self.connectionDialog)
        
        destFieldLabel = QtGui.QLabel(self.tr('Target Field'), self.connectionDialog)
        self.destFieldComboBox = QtGui.QComboBox(self.connectionDialog)

        self.sourceObjText = QtGui.QLineEdit(self.connectionDialog)
        self.destObjText = QtGui.QLineEdit(self.connectionDialog)

        okButton = QtGui.QPushButton(self.tr('OK'), self.connectionDialog)
        okButton.clicked.connect(self.connectionDialog.accept)
        cancelButton = QtGui.QPushButton(self.tr('Cancel'), self.connectionDialog)
        cancelButton.clicked.connect(self.connectionDialog.reject)

        self.connectionDialog.accepted.connect(self.createConnection)
        self.connectionDialog.rejected.connect(self.cancelConnection)

        layout = QtGui.QGridLayout()
        layout.addWidget(self.sourceTree, 0, 0, 5, 2)
        layout.addWidget(self.sourceObjText, 6, 0)
        layout.addWidget(sourceFieldLabel, 6, 1)
        layout.addWidget(self.sourceFieldComboBox, 6, 2)
        sep = QtGui.QFrame(self.connectionDialog)
        sep.setFrameStyle(QtGui.QFrame.VLine | QtGui.QFrame.Sunken)
        layout.addWidget(sep, 0, 3, -1, 1)
        layout.addWidget(self.destTree, 0, 4, 5, 2)
        layout.addWidget(self.destObjText, 6, 4)
        layout.addWidget(destFieldLabel, 6, 5)
        layout.addWidget(self.destFieldComboBox, 6, 6)
        
        layout.addWidget(cancelButton, 8, 0)
        layout.addWidget(okButton, 8, 4)

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

    def quit(self):
        """Do cleanup, saving, etc. before quitting."""
        QtGui.qApp.closeAllWIndows()

    def showRightBottomDocks(self, checked):
        """Hides the widgets on right and bottom dock area"""
        for child in self.findChildren(QtGui.QDockWidget):
            area = self.dockWidgetArea(child)
            if ( area == QtCore.Qt.BottomDockWidgetArea) or \
                    (area == QtCore.Qt.RightDockWidgetArea):
                child.setVisible(checked)

    def insertMooseObjectSlot(self, item):
        """Create an object of class specified by item and insert it
        as a child of currently selected element in the MOOSE element
        tree"""
        className = item.text()
        self.modelTreeWidget.insertMooseObjectSlot(className)

    def makeShellDock(self, interpreter=None, mode=MooseGlobals.CMD_MODE_PYMOOSE):
        """A MOOSE command line for GENESIS/Python interaction"""
        self.commandLineDock = QtGui.QDockWidget(self.tr('MOOSE Shell'), self)
        self.commandLineDock.setObjectName(self.tr('MooseShell'))
        self.shellWidget = MooseShell(interpreter)
        self.commandLineDock.setWidget(self.shellWidget)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.commandLineDock)
        self.commandLineDock.setObjectName('MooseCommandLine')
        return self.commandLineDock
        
    def makeObjectFieldEditor(self, item, column):
        """Creates a table-editor for a selected object."""
        obj = item.getMooseObject()
        try:
            self.objFieldEditModel = self.objFieldEditorMap[obj.path]
        except KeyError:
            config.LOGGER.debug('No editor for this object: %s' % (obj.path))
            self.objFieldEditModel = ObjectFieldsModel(obj)
            self.objFieldEditorMap[obj.path] = self.objFieldEditModel
            self.connect(self.objFieldEditModel, QtCore.SIGNAL('plotWindowChanged(const QString&, const QString&)'), self.changeFieldPlotWidget)
            if  hasattr(self, 'objFieldEditPanel'):
                self.objFieldEditPanel.setWindowTitle(self.tr(obj.name))
            else:
                self.objFieldEditPanel = QtGui.QDockWidget(self.tr(obj.name), self)
                self.objFieldEditPanel.setObjectName(self.tr('MooseObjectFieldEdit'))
                self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.objFieldEditPanel)
        self.objFieldEditor = QtGui.QTableView(self.objFieldEditPanel)            
        self.objFieldEditor.setModel(self.objFieldEditModel)
        self.objFieldEditModel.plotNames += [plot.objectName() for plot in self.plots]
        self.objFieldEditor.setItemDelegate(ObjectEditDelegate(self))
        self.connect(self.objFieldEditModel, 
                     QtCore.SIGNAL('objectNameChanged(const QString&)'),
                     item.updateSlot)
        # self.objFieldEditor.setContextMenuPolicy(Qt.CustomContextMenu)
        # self.connect(self.objFieldEditor, QtCore.SIGNAL('customContextMenuRequested ( const QPoint&)'), self.popupFieldMenu)
        self.objFieldEditPanel.setWidget(self.objFieldEditor)
	self.objFieldEditPanel.show()

    def createGLCellWidget(self):
    	"""Create a GLCell object to show the currently selected cell"""
        raise DeprecationWarning('This function is not implemented properly and is deprecated.')
        cellItem = self.modelTreeWidget.currentItem()
        cell = cellItem.getMooseObject()
        if not cell.className == 'Cell':
            QtGui.QMessageBox.information(self, self.tr('Incorrect type for GLCell'), self.tr('GLCell is for visualizing a cell. Please select one in the Tree view. Currently selected item is of ' + cell.className + ' class. Hover mouse over an item to see its class.'))
            return
        
    def createActions(self):
        # Actions for view menu
        # The following actions are to toggle visibility of various widgets
        self.glClientAction = self.glClientDock.toggleViewAction()
        self.glClientAction.setChecked(False)
        self.mooseTreeAction = self.mooseTreePanel.toggleViewAction()
        self.refreshMooseTreeAction = QtGui.QAction(self.tr('Refresh model tree'), self, triggered=self.modelTreeWidget.recreateTree)
        self.mooseClassesAction = self.mooseClassesPanel.toggleViewAction()
        self.mooseShellAction = self.commandLineDock.toggleViewAction()
        self.mooseShellAction.setChecked(False)
        self.mooseGLCellAction = QtGui.QAction(self.tr('GLCell'), self)
        self.mooseGLCellAction.setChecked(False)
        self.connect(self.mooseGLCellAction, QtCore.SIGNAL('triggered()'), self.createGLCellWidget)

        self.autoHideAction = QtGui.QAction(self.tr('Autohide during simulation'), self, checkable=True)
        self.autoHideAction.setChecked(self.settings.value(config.KEY_RUNTIME_AUTOHIDE).toBool())

        self.showRightBottomDocksAction = QtGui.QAction(self.tr('Right and Bottom Docks'), self, checkable=True, triggered=self.showRightBottomDocks)
        self.showRightBottomDocksAction.setChecked(False)

        self.tilePlotWindowsAction = QtGui.QAction(self.tr('Tile Plots'), self, checkable=True, triggered=self.centralPanel.tileSubWindows)
        self.cascadePlotWindowsAction = QtGui.QAction(self.tr('Cascade Plots'), self, checkable=True, triggered=self.centralPanel.cascadeSubWindows)
        
        self.subWindowLayoutActionGroup = QtGui.QActionGroup(self)
        self.subWindowLayoutActionGroup.addAction(self.tilePlotWindowsAction)
        self.subWindowLayoutActionGroup.addAction(self.cascadePlotWindowsAction)
        self.subWindowLayoutActionGroup.setExclusive(True)
        self.tilePlotWindowsAction.setChecked(True)

        # Action to configure plots
        self.configurePlotAction = QtGui.QAction(self.tr('Configure selected plots'), self, triggered=self.configurePlots)
        
        # Action to create connections
        self.connectionDialogAction = QtGui.QAction(self.tr('&Connect elements'), self, triggered=self.makeConnectionPopup)

        # Actions for file menu
        self.loadModelAction = QtGui.QAction(self.tr('Load Model'), self)
        self.connect(self.loadModelAction, QtCore.SIGNAL('triggered()'), self.popupLoadModelDialog)
        
        self.newPlotWindowAction = QtGui.QAction(self.tr('New Plot Window'), self)
        self.connect(self.newPlotWindowAction, QtCore.SIGNAL('triggered(bool)'), self.addPlotWindow)
        self.firstTimeWizardAction = QtGui.QAction(self.tr('FirstTime Configuration WIzard'), self)
        self.firstTimeWizardAction.triggered.connect(self.startFirstTimeWizard)

        # Actions to switch the command line between python and genesis mode.
        self.shellModeActionGroup = QtGui.QActionGroup(self)
        self.pythonModeAction = QtGui.QAction(self.tr('Python'), self.shellModeActionGroup, checkable=True)
        self.pythonModeAction.setChecked(True)
        self.genesisModeAction = QtGui.QAction(self.tr('GENESIS'), self.shellModeActionGroup, checkable=True)
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
        self.resetSettingsAction = QtGui.QAction(self.tr('Reset Settings'), self)
        self.connect(self.resetSettingsAction, QtCore.SIGNAL('triggered()'), self.resetSettings)
        self.showDocAction = QtGui.QAction(self.tr('Documentation'), self)
        self.contextHelpAction = QtGui.QAction(self.tr('Context Help'), self)
        self.runAction = QtGui.QAction(self.tr('Run Simulation'), self, triggered=self.runSlot)
        self.resetAction = QtGui.QAction(self.tr('Reset Simulation'), self, triggered=self.resetSlot)

        
        
        
    def runSquidDemo(self):
        path = self.demosDir + '/squid/qtSquid.py'
        subprocess.call(['python', path])

    def runIzhikevichDemo(self):
        path = self.demosDir + '/izhikevich/demogui_qt.py'
        subprocess.call(['python', path])

    def runGLCellDemo(self):
        path = self.demosDir + '/gl/glcelldemo.py'
        subprocess.call(['python', path], cwd=os.path.dirname(path))

    def runGLViewDemo(self):
        path = self.demosDir + '/gl/glviewdemo.py'
        subprocess.call(['python', path], cwd=os.path.dirname(path))
        
    
    def makeDemosMenu(self):
        self.squidDemoAction = QtGui.QAction(self.tr('Squid Axon'), self)
        self.connect(self.squidDemoAction, QtCore.SIGNAL('triggered()'), self.runSquidDemo)
        self.IzhikevichDemoAction = QtGui.QAction(self.tr('Izhikevich Model'), self)
        self.connect(self.IzhikevichDemoAction, QtCore.SIGNAL('triggered()'), self.runIzhikevichDemo)
        self.glCellDemoAction = QtGui.QAction(self.tr('GL Cell'), self)
        self.glCellDemoAction.triggered.connect(self.runGLCellDemo)
        self.glViewDemoAction = QtGui.QAction(self.tr('GL View'), self)
        self.glViewDemoAction.triggered.connect(self.runGLCellDemo)
        menu = QtGui.QMenu('&Demos and Tutorials', self)
        menu.addAction(self.squidDemoAction)
        menu.addAction(self.IzhikevichDemoAction)
        menu.addAction(self.glCellDemoAction)
        menu.addAction(self.glViewDemoAction)
        return menu
        
    def makeMenu(self):
        self.fileMenu = QtGui.QMenu(self.tr('&File'), self)
        self.fileMenu.addAction(self.newPlotWindowAction)
        self.fileMenu.addAction(self.loadModelAction)
        self.shellModeMenu = self.fileMenu.addMenu(self.tr('Moose Shell mode'))
        self.shellModeMenu.addActions(self.shellModeActionGroup.actions())
        self.fileMenu.addAction(self.firstTimeWizardAction)
        self.fileMenu.addAction(self.resetSettingsAction)
        self.fileMenu.addAction(self.quitAction)

        self.viewMenu = QtGui.QMenu('&View', self)
        self.viewMenu.addAction(self.glClientAction)
        self.viewMenu.addAction(self.mooseTreeAction)
        self.viewMenu.addAction(self.refreshMooseTreeAction)
        self.viewMenu.addAction(self.mooseClassesAction)
        self.viewMenu.addAction(self.mooseShellAction)
        self.viewMenu.addAction(self.autoHideAction)
        self.viewMenu.addAction(self.showRightBottomDocksAction)
        self.viewMenu.addSeparator().setText(self.tr('Layout Plot Windows'))
        self.viewMenu.addAction(self.tilePlotWindowsAction)
        self.viewMenu.addAction(self.cascadePlotWindowsAction)

        self.runMenu = QtGui.QMenu(self.tr('&Run'), self)
        self.runMenu.addAction(self.resetAction)
        self.runMenu.addAction(self.runAction)


        self.editModelMenu = QtGui.QMenu(self.tr('&Edit Model'), self)
        self.editModelMenu.addAction(self.connectionDialogAction)

        self.plotMenu = QtGui.QMenu(self.tr('&Plot Settings'), self)
        self.plotMenu.addAction(self.configurePlotAction)

        self.glMenu = QtGui.QMenu(self.tr('Open&GL'), self)
        self.glMenu.addAction(self.startGLWizardAction)
        self.glMenu.addAction(self.stopGLAction)
                                  
        self.helpMenu = QtGui.QMenu('&Help', self)
        self.helpMenu.addAction(self.showDocAction)
        self.helpMenu.addAction(self.contextHelpAction)
        self.demosMenu = self.makeDemosMenu()
        self.helpMenu.addMenu(self.demosMenu)
        self.helpMenu.addAction(self.aboutMooseAction)
        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.editModelMenu)
        self.menuBar().addMenu(self.runMenu)
        self.menuBar().addMenu(self.plotMenu)
        self.menuBar().addMenu(self.glMenu)
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
        geo_data = self.saveGeometry()
        layout_data = self.saveState()
        self.settings.setValue(config.KEY_WINDOW_GEOMETRY, QtCore.QVariant(geo_data))
        self.settings.setValue(config.KEY_WINDOW_LAYOUT, QtCore.QVariant(layout_data))
        self.settings.setValue(config.KEY_RUNTIME_AUTOHIDE, QtCore.QVariant(self.autoHideAction.isChecked()))
        self.settings.setValue(config.KEY_DEMOS_DIR, QtCore.QVariant(self.demosDir))
                              

    def loadLayout(self):
        '''Load the window layout.'''
        geo_data = self.settings.value(config.KEY_WINDOW_GEOMETRY).toByteArray()
        layout_data = self.settings.value(config.KEY_WINDOW_LAYOUT).toByteArray()
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
        if self.glClientDock.isHidden():
            # print 'Glclient is hidden'
            self.glClientAction.setChecked(False)
        else:
            # print 'Glclient is visible'
            self.glClientAction.setChecked(True)
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
	self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.mooseClassesPanel)
	self.mooseClassesWidget = makeClassList(self.mooseClassesPanel)
	self.mooseClassesPanel.setWidget(self.mooseClassesWidget)
        config.LOGGER.debug('createMooseClassesPanel - end')

    def createMooseTreePanel(self):
        config.LOGGER.debug('createMooseTreePanel - start')
	self.mooseTreePanel = QtGui.QDockWidget(self.tr('Element Tree'), self)
        self.mooseTreePanel.setObjectName(self.tr('MooseClassPanel'))
	self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.mooseTreePanel)
	self.modelTreeWidget = MooseTreeWidget(self.mooseTreePanel) 
        self.modelTreeWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.connect(self.modelTreeWidget, QtCore.SIGNAL('customContextMenuRequested ( const QPoint&)'), self.createConnectionMenu)
	self.mooseTreePanel.setWidget(self.modelTreeWidget)
        config.LOGGER.debug('createMooseTreePanel - end')
        
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
        self.runtimeText = QtGui.QLineEdit('%1.3e' % (MooseHandler.runtime), self.controlPanel)
        self.updateTimeLabel = QtGui.QLabel(self.tr('Update interval for plots (second):'), self.controlPanel)
        self.updateTimeText = QtGui.QLineEdit('%1.3e' % (MooseHandler.plotupdate_dt), self.controlPanel)
        self.resetButton = QtGui.QPushButton(self.tr('Reset'), self.controlPanel)
        self.runButton = QtGui.QPushButton(self.tr('Run'), self.controlPanel)
        self.simdtLabel = QtGui.QLabel(self.tr('Simulation timestep (second):'), self.controlPanel)
        self.plotdtLabel = QtGui.QLabel(self.tr('Plotting timestep (second):'), self.controlPanel)
        self.gldtLabel = QtGui.QLabel(self.tr('3D visualization timestep (second):'), self.controlPanel)
        self.simdtText = QtGui.QLineEdit('%1.3e' % (MooseHandler.simdt), self.controlPanel)
        self.plotdtText = QtGui.QLineEdit('%1.3e' % (MooseHandler.plotdt), self)
        self.gldtText = QtGui.QLineEdit('%1.3e' % (MooseHandler.gldt), self.controlPanel)
        self.overlayCheckBox = QtGui.QCheckBox(self.tr('Overlay plots'), self.controlPanel)
        
        self.connect(self.runButton, QtCore.SIGNAL('clicked()'), self.runSlot)
        self.connect(self.resetButton, QtCore.SIGNAL('clicked()'), self.resetSlot)
        layout.addWidget(self.simdtLabel, 0,0)
        layout.addWidget(self.simdtText, 0, 1)
        layout.addWidget(self.plotdtLabel, 1, 0)
        layout.addWidget(self.plotdtText, 1, 1)
        layout.addWidget(self.gldtLabel, 2, 0)
        layout.addWidget(self.gldtText, 2, 1)
        layout.addWidget(self.overlayCheckBox, 3, 0)
        layout.addWidget(self.runtimeLabel, 4, 0)
        layout.addWidget(self.runtimeText, 4, 1)
        layout.addWidget(self.updateTimeLabel, 5, 0)
        layout.addWidget(self.updateTimeText, 5,1)
        layout.addWidget(self.resetButton, 6, 0)
        layout.addWidget(self.runButton, 6, 1)
        self.controlPanel.setLayout(layout)
        self.controlDock.setWidget(self.controlPanel)

    def addPlotWindow(self):
        title = self.tr('Plot %d' % (len(self.plots)))
        plotWindow = QtGui.QMainWindow()
        plotWindow.setWindowTitle(title)
        plot = MoosePlot(plotWindow)
        plot.setObjectName(title)
        plotWindow.setCentralWidget(plot)
        self.plots.append(plot)
        subWindow = self.centralPanel.addSubWindow(plotWindow)
        self.centralPanel.setActiveSubWindow(subWindow)
        plotWindow.show()
        if hasattr(self, 'cascadePlotWindowsAction') and self.cascadePlotWindowsAction.isChecked():
            self.centralPanel.cascadeSubWindows()
        else:
            self.centralPanel.tileSubWindows()
        self.currentPlotWindow = plotWindow
        return plotWindow
        

    def popupLoadModelDialog(self):
        fileDialog = QtGui.QFileDialog(self)
        fileDialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        ffilter = ''
        for key in sorted(self.mooseHandler.fileExtensionMap.keys()):
            ffilter = ffilter + key + ';;'
        ffilter = ffilter[:-2]
        fileDialog.setFilter(self.tr(ffilter))
        # The following version gymnastic is because QFileDialog.selectNameFilter() was introduced in Qt 4.4
        qtVersion = str(QtCore.QT_VERSION_STR).split('.')
        major = int(qtVersion[0])
        minor = int(qtVersion[1])
        if (major > 4)or ((major == 4) and (minor >= 4)):
            for key, value in self.mooseHandler.fileExtensionMap.items():
                if value == MooseHandler.type_genesis:
                    fileDialog.selectNameFilter(key)
                    break
	if fileDialog.exec_():
	    fileNames = fileDialog.selectedFiles()
	    fileFilter = fileDialog.selectedFilter()
	    fileType = self.mooseHandler.fileExtensionMap[str(fileFilter)]
	    directory = fileDialog.directory() # Potential bug: if user types the whole file path, does it work? - no but gives error message
	    for fileName in fileNames: 
                modeltype  = self.mooseHandler.loadModel(str(fileName), str(fileType))
		if modeltype == MooseHandler.type_kkit:
                    self.populateKKitPlots()
                print 'Loaded model',  fileName, 'of type', modeltype
            self.modelTreeWidget.recreateTree()


    def resetSettings(self):
        self.settings.clear()

    def setCurrentElement(self, item, column):
        """Set the current object of the mooseHandler"""
        current_element = item.getMooseObject()
        self.mooseHandler._current_element = current_element


    def resetSlot(self):
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
        try:            
            gldt = float(str(self.gldtText.text()))
        except ValueError, gldt_err:
            print 'Error setting 3D visualization time step:', gldt_err
            gldt = MooseHandler.gldt
            self.gldtText.setText('%1.3e' % (MooseHandler.gldt))
        try:
            updateInterval = float(str(self.updateTimeText.text()))
            if updateInterval < 0.0:
                updateInterval = MooseHandler.plotupdate_dt
        except ValueError:
            updateInterval = MooseHandler.plotupdate_dt
            self.updateTimeText.setText(str(updateInterval))
        self.mooseHandler.doReset(simdt, plotdt, gldt, updateInterval)
        for table, plot in self.tablePlotMap.items():
            table.clear()
            plot.setOverlay(self.overlayCheckBox.isChecked())
            plot.reset()
                    

    def runSlot(self):
        """Run the simulation.

        """
        if self.autoHideAction.isChecked():
            if self.commandLineDock.isVisible():
                self.commandLineDock.setVisible(False)
            if self.mooseClassesPanel.isVisible():
                self.mooseClassesPanel.setVisible(False)
            if self.glClientDock.isVisible():
                self.glClientDock.setVisible(False)
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
            plotName = activePlot.windowTitle() # The window title is the plot name
            # print 'configurePlots', plotName 
            for plot in self.plots:
                if plot.objectName() == plotName:
                    plot.reconfigureSelectedCurves(pen, symbol, style, attribute)
                    break

    def createConnection(self):
        if (self._srcElement is None) or (self._destElement is None):
            return
        src = self._srcElement.path + '/' + str(self.sourceFieldComboBox.currentText())
        dest = self._destElement.path + '/' + str(self.destFieldComboBox.currentText())
        self.mooseHandler.setConnSrc(src)
        self.mooseHandler.setConnDest(dest)
        ret = self.mooseHandler.doConnect()
        print 'Connecting %s to %s: %s' % (src, dest, 'success' if ret else 'failed')        
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

    def startFirstTimeWizard(self):
        firstTimeWizard = FirstTimeWizard(self)
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
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    icon = QtGui.QIcon('moose_icon.png')
    app.setWindowIcon(icon)

    QtCore.QObject.connect(app, QtCore.SIGNAL('lastWindowClosed()'), app, QtCore.SLOT('quit()'))
    if not config.get_settings().contains(config.KEY_FIRSTTIME):
        firstTimeWizard = FirstTimeWizard()
        firstTimeWizard.setModal(QtCore.Qt.ApplicationModal)
        firstTimeWizard.show()
    mainWin = MainWindow()
    mainWin.show()
    app.exec_()
	
