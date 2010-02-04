# moosegui.py --- 
# 
# Filename: moosegui.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Wed Jan 20 15:24:05 2010 (+0530)
# Version: 
# Last-Updated: Thu Feb  4 14:00:11 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 667
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

# The following line is for ease in development environment. Normal
# users will have moose.py and _moose.so installed in some directory
# in PYTHONPATH.  If you have these two files in /usr/local/moose, you
# can enter the command:
#
# "export PYTHONPATH=$PYTHONPATH:/usr/local/moose" 
#
# in the command prompt before running the
# moosegui with "python moosegui.py"
sys.path.append('/home/subha/src/moose/pymoose')



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
        aboutText = '<html><h3>%s</h3><p>%s</p><p>%s</p><p>%s</p></html>' % (MooseGlobals.TITLE_TEXT, MooseGlobals.COPYRIGHT_TEXT, MooseGlobals.LICENSE_TEXT, MooseGlobals.ABOUT_TEXT)
        aboutMooseLabel = QtGui.QLabel(parent)
        aboutMooseLabel.setText(aboutText)
        aboutMooseLabel.setWordWrap(True)
        aboutMooseLabel.setAlignment(Qt.AlignHCenter)
        aboutMooseLabel.setSizePolicy(sizePolicy)
        return aboutMooseLabel
    
class MainWindow(QtGui.QMainWindow):
            
    def __init__(self, interpreter=None, parent=None):
        print 'MainWindow.__init__'
	QtGui.QMainWindow.__init__(self, parent)
        print 'QMainWindow.__init__(self, *args)'
        self.layoutFile = 'moose.layout'
	self.resize(800, 600)
        self.setDockOptions(self.AllowNestedDocks | self.AllowTabbedDocks | self.ForceTabbedDocks | self.AnimatedDocks)

        # This is the element tree of MOOSE
	self.mooseTreePanel = QtGui.QDockWidget(self.tr('Element Tree'), self)
        self.mooseTreePanel.setObjectName(self.tr('MooseClassPanel'))
	self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.mooseTreePanel)
	self.modelTreeWidget = makeModelTree(self.mooseTreePanel) 
	self.mooseTreePanel.setWidget(self.modelTreeWidget)
        # List of classes - one can double click on any class to
        # create an instance under the currently selected element in
        # mooseTreePanel
	self.mooseClassesPanel = QtGui.QDockWidget(self.tr('Classes'), self)
        self.mooseClassesPanel.setObjectName(self.tr('MooseClassPanel'))
	self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.mooseClassesPanel)
	self.mooseClassesWidget = makeClassList(self.mooseClassesPanel)
	self.mooseClassesPanel.setWidget(self.mooseClassesWidget)
        
        # Connect the double-click event on modelTreeWidget items to
        # creation of the object editor.
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
        self.objFieldEditPanel.setWidget(self.objFieldEditor)
	self.objFieldEditPanel.show()

    def makeMenu(self):
        pass

    def saveLayout(self):
        """Overriding QMainWindow.close() to save window state."""

        stateFile = QtCore.QFile(self.layoutFile)
        if not stateFile.open(QtCore.QFile.WriteOnly):
            message = self.tr('Failed to open %1\n%2').arg(self.layoutFile).arg(stateFile.errorString())
            messageBox = QtGui.QMessageBox.warning(self, self.tr('Error'), message)
            return
        geo_data = self.saveGeometry()
        layout_data = self.saveState()
        ok = stateFile.putChar(chr(geo_data.size()))
        if ok:
            ok = stateFile.write(geo_data) == geo_data.size()
        if ok:
            ok = stateFile.write(layout_data) == layout_data.size()
            
        if not ok:
            msg = self.tr("Error writing to %1\n%2").arg(self.layoutFile).arg(stateFile.errorString())
            QtGui.QMessageBox.warning(self, self.tr("Error"), msg)
            stateFile.close()
            stateFile.remove()
            return
        else:
            stateFile.close()

    def loadLayout(self):
        stateFile = QtCore.QFile(self.layoutFile)
        ok = stateFile.open(QtCore.QFile.ReadOnly)
        if not ok:
            print 'Will save layout info for the first time at exit.'
            stateFile.close()
            return 
        geo_size = QtCore.QChar()
        (ok, geo_size) = stateFile.getChar()
        if ok:
            geo_data = stateFile.read(ord(geo_size))
            ok = len(geo_data) == ord(geo_size)
        if ok:
            layout_data = stateFile.readAll()
            ok = len(layout_data) > 0
        if ok:
            ok = self.restoreGeometry(geo_data)
        if ok:
            ok = self.restoreState(layout_data)
        if not ok:
            msg = self.tr('Error reading %1').arg(self.layoutFile)
            QtGui.QMessageBox.warning(self, self.tr('Error'), msg)
            stateFile.close()
            return
        stateFile.close()
    
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    QtCore.QObject.connect(app, QtCore.SIGNAL('lastWindowClosed()'), app, QtCore.SLOT('quit()'))
    mainWin = MainWindow()
    mainWin.show()
    app.exec_()
	



# 
# moosegui.py ends here
