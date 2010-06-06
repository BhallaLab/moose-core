# moosegui.py --- 
# 
# Filename: moosegui.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Wed Jan 20 15:24:05 2010 (+0530)
# Version: 
# Last-Updated: Mon Jun  7 00:54:52 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 984
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
import code
from datetime import date

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
from objedit import ObjectFieldsModel
from moosetree import *
from mooseclasses import *
from mooseglobals import MooseGlobals
from mooseshell import MooseShell

def makeModelTree(parent):
    mooseTree = MooseTreeWidget(parent)
    return mooseTree

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


def makeAboutMooseLabel(parent):
        """Create a QLabel with basic info about MOOSE."""
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        aboutText = '<html><h3>%s</h3><p>%s</p><p>%s</p><p>%s</p></html>' % \
            (MooseGlobals.TITLE_TEXT, 
             MooseGlobals.COPYRIGHT_TEXT, 
             MooseGlobals.LICENSE_TEXT, 
             MooseGlobals.ABOUT_TEXT)

        aboutMooseLabel = QtGui.QLabel(parent)
        aboutMooseLabel.setText(aboutText)
        aboutMooseLabel.setWordWrap(True)
        aboutMooseLabel.setAlignment(Qt.AlignHCenter)
        aboutMooseLabel.setSizePolicy(sizePolicy)
        return aboutMooseLabel
    
class MainWindow(QtGui.QMainWindow):
            
    def __init__(self, interpreter=None, parent=None):
	QtGui.QMainWindow.__init__(self, parent)
        self.settings = config.get_settings()
        self.resize(800, 600)
        self.setDockOptions(self.AllowNestedDocks | self.AllowTabbedDocks | self.ForceTabbedDocks | self.AnimatedDocks)
        
        # This is the element tree of MOOSE
        self.createMooseTreePanel()
        # List of classes - one can double click on any class to
        # create an instance under the currently selected element in
        # mooseTreePanel
        self.createMooseClassesPanel()

        # Create a widget to configure the glclient
        self.createGLClientDock()
        # Connect the double-click event on modelTreeWidget items to
        # creation of the object editor.
        # TODO - will create a right-click menu
        self.connect(self.modelTreeWidget, 
                     QtCore.SIGNAL('itemDoubleClicked(QTreeWidgetItem *, int)'),
                     self.makeObjectFieldEditor)
        # We use the objFieldEditorMap to store reference to cache the
        # model objects for moose objects.
        self.objFieldEditorMap = {}
        self.makeShellDock(interpreter)
        # By default, we show information about MOOSE in the central widget
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        self.aboutMooseLabel = makeAboutMooseLabel(self)
        self.aboutMooseLabel.setSizePolicy(sizePolicy)
        self.setCentralWidget(self.aboutMooseLabel)
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


    def quit(self):
        """Do cleanup, saving, etc. before quitting."""
        QtGui.qApp.closeAllWIndows()

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
            print 'Key Error'
            self.objFieldEditModel = ObjectFieldsModel(obj)
            self.objFieldEditorMap[obj.path] = self.objFieldEditModel
            if  hasattr(self, 'objFieldEditPanel'):
                self.objFieldEditPanel.setWindowTitle(self.tr(obj.name))
            else:
                self.objFieldEditPanel = QtGui.QDockWidget(self.tr(obj.name), self)
                self.objFieldEditPanel.setObjectName(self.tr('MooseObjectFieldEdit'))
                self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.objFieldEditPanel)
        self.objFieldEditor = QtGui.QTableView(self.objFieldEditPanel)            
        self.objFieldEditor.setModel(self.objFieldEditModel)        
        self.connect(self.objFieldEditModel, 
                     QtCore.SIGNAL('objectNameChanged(const QString&)'),
                     item.updateSlot)
        self.objFieldEditor.setContextMenuPolicy(Qt.CustomContextMenu)
        self.connect(self.objFieldEditor, QtCore.SIGNAL('customContextMenuRequested ( const QPoint&)'), self.popupFieldMenu)
        self.objFieldEditPanel.setWidget(self.objFieldEditor)
	self.objFieldEditPanel.show()

    def createGLCellWidget(self):
        '''Create a GLCell object to show the currently selected cell'''
        cellItem = self.modelTreeWidget.currentItem()
        cell = cellItem.getMooseObject()
        if not cell.className == 'Cell':
            QtGui.QMessageBox.information(self, self.tr('Incorrect type for GLCell'), self.tr('GLCell is for visualizing a cell. Please select one in the Tree view. Currently selected item is of ' + cell.className + ' class. Hover mouse over an item to see its class.'))
            return

    def createActions(self):
        self.glClientAction = self.glClientDock.toggleViewAction()
        self.mooseTreeAction = self.mooseTreePanel.toggleViewAction()
        self.mooseClassesAction = self.mooseClassesPanel.toggleViewAction()
        self.mooseShellAction = self.commandLineDock.toggleViewAction()
        self.mooseGLCellAction = QtGui.QAction(self.tr('GLCell'), self)
        self.connect(self.mooseGLCellAction, QtCore.SIGNAL('triggered()'), self.createGLCellWidget)
        self.quitAction = QtGui.QAction(self.tr('&Quit'), self)
        self.quitAction.setShortcut(QtGui.QKeySequence(self.tr('Ctrl+Q')))
        self.connect(self.quitAction, QtCore.SIGNAL('triggered()'), QtGui.qApp, QtCore.SLOT('closeAllWindows()'))
        self.aboutMooseAction = QtGui.QAction(self.tr('&About'), self)
        self.connect(self.aboutMooseAction, QtCore.SIGNAL('triggered()'), makeAboutMooseLabel)
        self.resetSettingsAction = QtGui.QAction(self.tr('Reset Settings'), self)
        self.connect(self.resetSettingsAction, QtCore.SIGNAL('triggered()'), self.resetSettings)
        # TODO: the following actions are yet to be implemented.
        self.showDocAction = QtGui.QAction(self.tr('Documentation'), self)
        self.contextHelpAction = QtGui.QAction(self.tr('Context Help'), self)
        
    def runSquidDemo(self):
        QtGui.QMessageBox.information(self, 'Not yet incorporated', 'this demo is yet to be incorporated into moosegui')

    def runIzhikevichDemo(self):
        QtGui.QMessageBox.information(self, 'Not yet incorporated', 'this demo is yet to be incorporated into moosegui')
    
    def makeDemosMenu(self):
        self.squidDemoAction = QtGui.QAction(self.tr('Squid Axon'), self)
        self.connect(self.squidDemoAction, QtCore.SIGNAL('triggered()'), self.runSquidDemo)
        self.IzhikevichDemoAction = QtGui.QAction(self.tr('Izhikevich Model'), self)
        self.connect(self.IzhikevichDemoAction, QtCore.SIGNAL('triggered()'), self.runIzhikevichDemo)
        menu = QtGui.QMenu('&Demos and Tutorials', self)
        menu.addAction(self.squidDemoAction)
        menu.addAction(self.IzhikevichDemoAction)
        return menu
        # TODO: create a class for the demos menu.
        
    def makeMenu(self):
        self.fileMenu = QtGui.QMenu('&File', self)
        self.fileMenu.addAction(self.quitAction)
        self.fileMenu.addAction(self.resetSettingsAction)
        self.viewMenu = QtGui.QMenu('&View', self)
        self.viewMenu.addAction(self.glClientAction)
        self.viewMenu.addAction(self.mooseTreeAction)
        self.viewMenu.addAction(self.mooseClassesAction)
        self.viewMenu.addAction(self.mooseShellAction)
        self.helpMenu = QtGui.QMenu('&Help', self)
        # TODO: code the actual functions
        self.helpMenu.addAction(self.showDocAction)
        self.helpMenu.addAction(self.contextHelpAction)
 
        self.demosMenu = self.makeDemosMenu()
        self.helpMenu.addMenu(self.demosMenu)
        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)

    def saveLayout(self):
        '''Save window layout'''
        geo_data = self.saveGeometry()
        layout_data = self.saveState()
        self.settings.setValue(config.KEY_WINDOW_GEOMETRY, QtCore.QVariant(geo_data))
        self.settings.setValue(config.KEY_WINDOW_LAYOUT, QtCore.QVariant(layout_data))

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
            print 'Glclient is hidden'
            self.glClientAction.setChecked(False)
        else:
            print 'Glclient is visible'
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
	self.modelTreeWidget = makeModelTree(self.mooseTreePanel) 
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

    def popupFieldMenu(self, clickpoint):
        print 'PopupFieldMenu'
        index = self.objFieldEditor.indexAt(clickpoint)
        data = self.objFieldEditModel.data(self.objFieldEditModel.createIndex(index.row(), 0))
        print data
        menu = QtGui.QMenu(self.objFieldEditor)
        self.actionPlotField = menu.addAction('Plot this field')
        self.connect(self.actionPlotField, QtCore.SIGNAL('triggered()'), self.plotThisFieldSlot)
        menu.popup(self.objFieldEditor.mapToGlobal(clickpoint))

    def plotThisFieldSlot(self):
        config.LOGGER.debug('plotThisFieldSlot - start')
        config.LOGGER.warning('Not ported this functionality yet.')
        moose_object = self.modelTreeWidget.currentItem().getMooseObject()
        row = self.objFieldEditor.currentIndex().row()
        index = self.objFieldEditModel.createIndex(row, 0)
        print index.row(), index.column()
        field_name = self.objFieldEditModel.data(index)        
        # table = self.mooseHandler.createTableForMolecule(moose_object, field_name)
        # print table.path
        config.LOGGER.debug('plotThisFieldSlot - end')

    def resetSettings(self):
        self.settings.clear()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    QtCore.QObject.connect(app, QtCore.SIGNAL('lastWindowClosed()'), app, QtCore.SLOT('quit()'))
    mainWin = MainWindow()
    mainWin.show()
    app.exec_()
	



# 
# moosegui.py ends here
