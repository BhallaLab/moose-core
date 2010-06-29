# mainwin.py --- 
# 
# Filename: mainwin.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Tue Jun 16 11:38:46 2009 (+0530)
# Version: 
# Last-Updated: Tue Jun 29 12:22:06 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 884
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
#  2009-06-26 13:53:14 (+0530) This is right now a single threaded
#  app. Running the simulation in separate thread will be a lot of
#  work: all moose objects should live in a single thread and we'll
#  need to guard access to every moose variable from other threads
#  using locks. That will be a lot of work. Perticularly, the
#  updatePlots method will need to get a serialized copy of the moose
#  tables for thread safety.
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
import math
from collections import defaultdict
from moosetree import MooseTreeWidget
from moosepropedit import PropertyModel
from moosehandler import MHandler
from filetypeutil import FileTypeChecker
from mooseplots import MoosePlots
from settingsdialog import SettingsDialog

from PyQt4.Qt import Qt
from PyQt4 import QtCore, QtGui
from PyQt4 import Qwt5 as Qwt
import PyQt4.Qwt5.qplt as qplt
import PyQt4.Qwt5.anynumpy as numpy


from ui_mainwindow import Ui_MainWindow

class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    """Main Window for MOOSE GUI"""
    def __init__(self, loadFile=None, fileType=None):
	QtGui.QMainWindow.__init__(self)
	self.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(':moose_thumbnail.png'))
        self.settingsDialog = SettingsDialog()
        self.settingsDialog.hide()
        layout = QtGui.QVBoxLayout(self.modelTreeTab)
        self.modelTreeTab.setLayout(layout)
        layout.addWidget(self.modelTreeContainerWidget)
        self.modelTreeWidget.headerItem().setHidden(True)
        self.plotUpdateIntervalLabel.setText(self.tr('<br>Plot update interval (<b>steps</b>)'))
        self.runTimeLabel.setText(self.tr('<br>Run for (<b>seconds</b>)'))
#         self.modelTreeWidget.show()
#         self.mooseClassToolBox.show()
###################
#         self.plotsLayout = QtGui.QHBoxLayout(self.plotsGroupBox)
#         self.plotsGroupBox.setLayout(self.plotsLayout)
#         self.plotsScrollArea = QtGui.QScrollArea(self.plotsGroupBox)
#         self.plotsLayout.addWidget(self.plotsScrollArea)
#         self.plots = MoosePlots(self.plotsScrollArea)
#         self.plotsScrollArea.setWidget(self.plots)
#         self.plots.show()
##################
        self.plots = MoosePlots(self.plotsGroupBox)
        self.plotsLayout = QtGui.QHBoxLayout(self.plotsGroupBox)
        self.plotsGroupBox.setLayout(self.plotsLayout)
        self.plotsLayout.addWidget(self.plots)
        
######################
        self.plots.show()
        self.isModelLoaded = False
        self.stopFlag = False
	self.mooseHandler = MHandler()
        self.runTimeLineEdit.setText(self.tr(str(self.mooseHandler.runTime)))
        self.plotUpdateIntervalLineEdit.setText(self.tr(str(self.mooseHandler.updateInterval)))
	self.currentDirectory = '.'
	self.fileType = fileType
	self.fileTypes = MHandler.file_types
	self.filter = ''
	self.setupActions()
        if loadFile is not None and fileType is not None:
            self.load(loadFile, fileType)


    def setupActions(self):
	self.connect(self.actionQuit,
		     QtCore.SIGNAL('triggered()'), 
		     QtGui.qApp, 
		     QtCore.SLOT('closeAllWindows()'))		     
	self.connect(self.actionLoad, 
		     QtCore.SIGNAL('triggered()'),		   
		     self.loadFileDialog)
	self.connect(self.actionSquid_Axon,
		     QtCore.SIGNAL('triggered()'),
		     self.loadSquid_Axon_Tutorial)
	self.connect(self.actionIzhikevich_Neurons,
		     QtCore.SIGNAL('triggered()'),
		     self.loadIzhikevich_Neurons_Tutorial)
        self.connect(self.actionGLCell,
                     QtCore.SIGNAL('triggered()'),
                     self.loadGLCellDemo)
        self.connect(self.actionGLView,
                     QtCore.SIGNAL('triggered()'),
                     self.loadGLViewDemo)
	self.connect(self.actionAbout_MOOSE,
		     QtCore.SIGNAL('triggered()'),
		     self.showAbout_MOOSE)
	self.connect(self.actionReset,
		     QtCore.SIGNAL('triggered()'),
		     self.reset)
	self.connect(self.actionStart,
		     QtCore.SIGNAL('triggered()'),
		     self.run)
        self.connect(self.actionStop,
                     QtCore.SIGNAL('triggered()'),
                     self.stop)
        self.connect(self.runPushButton,
                     QtCore.SIGNAL('clicked()'),
                     self.run)
        self.connect(self.resetPushButton,
                     QtCore.SIGNAL('clicked()'),
                     self.reset)
#         self.connect(self.stopPushButton,
#                      QtCore.SIGNAL('clicked()'),
#                      self.mooseHandler.stop)
        self.connect(self.plotUpdateIntervalLineEdit,
                     QtCore.SIGNAL('editingFinished()'),
                     self.plotUpdateIntervalSlot)
        self.connect(self.modelTreeWidget, 
                     QtCore.SIGNAL('itemDoubleClicked(QTreeWidgetItem *, int)'),
                     self.popupPropertyEditor)
        self.connect(self.mooseHandler, 
                     QtCore.SIGNAL('updated()'), 
                     self.updatePlots)
        self.connect(self.runTimeLineEdit, 
                     QtCore.SIGNAL('editingFinished()'),
                     self.runTimeSlot)
        self.connect(self.rescalePlotsPushButton,
                     QtCore.SIGNAL('clicked()'),
                     self.plots.rescalePlots)

        self.connect(self.actionSettings,
                     QtCore.SIGNAL('triggered()'),
                     self.popupSettings)
        for listWidget in self.mooseClassToolBox.listWidgets:
            self.connect(listWidget, 
                         QtCore.SIGNAL('itemDoubleClicked(QListWidgetItem*)'), 
                         self.insertMooseObjectSlot)
        

    def popupPropertyEditor(self, item, column):
        """Pop-up a property editor to edit the Moose object in the item"""
        obj = item.getMooseObject()
        self.propertyModel = PropertyModel(obj)
        self.connect(self.propertyModel, 
                     QtCore.SIGNAL('objectNameChanged(const QString&)'),
                     item.updateSlot)
        self.propertyEditor = QtGui.QTableView()
        self.propertyEditor.setModel(self.propertyModel)
        self.propertyEditor.setContextMenuPolicy(Qt.CustomContextMenu)
        self.connect(self.propertyEditor, QtCore.SIGNAL('customContextMenuRequested ( const QPoint&)'), self.popupFieldMenu)
        self.propertyEditor.show()

    def loadFileDialog(self):
	fileDialog = QtGui.QFileDialog(self)
	fileDialog.setFileMode(QtGui.QFileDialog.ExistingFile)
	ffilter = ''
	for key in self.fileTypes.keys():
	    ffilter = ffilter + key + ';;'
	ffilter = ffilter[:-2]
	fileDialog.setFilter(self.tr(ffilter))
        # The following version gymnastic is because QFileDialog.selectNameFilter() was introduced in Qt 4.4
        qtVersion = str(QtCore.QT_VERSION_STR).split('.')
        major = int(qtVersion[0])
        minor = int(qtVersion[1])
        if (major > 4)or ((major == 4) and (minor >= 4)):
            for key, value in self.fileTypes.items():
                if value == FileTypeChecker.type_genesis:
                    fileDialog.selectNameFilter(key)
                    break
	if fileDialog.exec_():
	    fileNames = fileDialog.selectedFiles()
	    fileFilter = fileDialog.selectedFilter()
	    fileType = self.fileTypes[str(fileFilter)]
# 	    print 'file type:', fileType
	    directory = fileDialog.directory() # Potential bug: if user types the whole file path, does it work? - no but gives error message
	    for fileName in fileNames: 
# 		print fileName
		self.load(fileName, fileType)


    def loadIzhikevich_Neurons_Tutorial(self):
#     self.mooseHandler.load('../../DEMOS/pymoose/izhikevich/Izhikevich.py', 'MOOSE')
        spath = sys.path[0] + '/../../DEMOS/pymoose/izhikevich/demogui_qt.py'
        self.mooseHandler.load(spath, 'MOOSE')
        
    def loadGLCellDemo(self):
        spath = sys.path[0] + '/demo/glcelldemo.py'
        self.mooseHandler.load(spath, 'MOOSE')

    def loadGLViewDemo(self):
        spath = sys.path[0] + '/demo/glviewdemo.py'
        self.mooseHandler.load(spath, 'MOOSE')
        


    def loadSquid_Axon_Tutorial(self):
#     self.mooseHandler.load('../../DEMOS/pymoose/squid/qtSquid.py', 'MOOSE')
       spath = sys.path[0] + '/../../DEMOS/pymoose/squid/qtSquid.py'
       self.mooseHandler.load(spath, 'MOOSE')
       
	


    def showAbout_MOOSE(self):
	about = QtCore.QT_TR_NOOP('<p>MOOSE is the Multi-scale Object Oriented Simulation Environment.</p>'
				  '<p>It is a general purpose simulation environment for computational neuroscience and chemical kinetics.</p>'
				  '<p>Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS</p>'
				  '<p>It is made available under the terms of the GNU Lesser General Public License version 2.1. See the file COPYING.LIB for the full notice.</p>'
                  '<p>homepage: <a href="http://moose.ncbs.res.in">http://moose.ncbs.res.in</a></p>')
	aboutDialog = QtGui.QMessageBox.information(self, self.tr('About MOOSE'), about)

    def plotUpdateIntervalSlot(self):
        try:
            new_interval = int(str(self.plotUpdateIntervalLineEdit.text()))
            self.plotUpdateInterval = new_interval
        except ValueError:
            print 'Update steps must be whole number.'
            self.plotUpdateIntervalLineEdit.setText(self.tr(str(self.plotUpdateInterval)))
    
    def runTimeSlot(self):
        try:
            runtime = float(str(self.runTimeLineEdit.text()))
            self.mooseHandler.runTime = runtime
        except ValueError:
            print 'Error in converting runtime to float'
            self.runTimeLineEdit.setText(self.tr(str(self.mooseHandler.updateInterval)))
        
            
    def run(self):
        #print 'Going to run'
        if not self.isModelLoaded:
            errorDialog = QtGui.QErrorMessage(self)
            errorDialog.setModal(True)
            errorDialog.showMessage('<p>You must load a model first.</p>')
            return
        # The run for update interval and replot data until the full runtime has been executed
        # But this too is unresponsive
        self.mooseHandler.run()

    def updatePlots(self):        
        """Update the plots"""
        if self.plots is not None:
            self.plots.updatePlots(self.mooseHandler)
            self.update()
        self.currentTimeLabel.setText(self.tr('Current time (seconds): %g' %  self.mooseHandler.currentTime()))

    def reset(self):
        if not self.isModelLoaded:
            QtGui.QErrorMessage.showMessage('<p>You must load a model first.</p>')
        else:
            self.mooseHandler.reset()
            self.plots.resetPlots()
        self.currentTimeLabel.setText(self.tr('Current time (seconds): %g' % self.mooseHandler.currentTime()))
                
    
    def load(self, fileName, fileType):
 #      if self.isModelLoaded:            
#            import subprocess
#            subprocess.call(['python', 'main.py', fileName, fileType])
        fileType = FileTypeChecker(str(fileName)).fileType()
        print 'File is of type:', fileType
        self.mooseHandler.context.setCwe(self.modelTreeWidget.currentItem().getMooseObject().path)
        self.mooseHandler.load(fileName, fileType)
        self.isModelLoaded = True
        self.modelTreeWidget.recreateTree()
        if fileType is FileTypeChecker.type_sbml:
            self.moleculeListWidget = QtGui.QListWidget(self)
            self.moleculeItems = []
            for molecule in self.mooseHandler.moleculeList:
                item = QtGui.QListWidgetItem(self.moleculeListWidget)
                item.setText(molecule.name)
                item.setData(QtCore.Qt.UserRole, QtCore.QVariant(self.tr(molecule.path)))
                self.moleculeItems.append(item)
            self.simulationWidget.layout().addWidget(self.moleculeListWidget)
            self.connect(self.moleculeListWidget, 
                         QtCore.SIGNAL('itemDoubleClicked(QListWidgetItem*)'),
                         self.addMoleculeInPlot)
            
            self.moleculeListWidget.show()

        self.plots.loadPlots(self.mooseHandler, fileType)
        self.currentTimeLabel.setText(self.tr('Current time (seonds): %g' % self.mooseHandler.currentTime()))
#         self.update()

    def addMoleculeInPlot(self, item):
        self.mooseHandler.createTableForMolecule(str(item.text()))
        self.plots.loadPlots(self.mooseHandler, FileTypeChecker.type_sbml)
        self.plots.setVisible(True)
        self.modelTreeWidget.recreateTree()
        self.update()

    # Until MOOSE has a way of getting stop command from outside
    def stop(self):
        self.mooseHandler.stop()

    def insertMooseObjectSlot(self, item):
        self.modelTreeWidget.insertMooseObjectSlot(item.text())

    def popupSettings(self):
        self.settingsDialog.show()
        result = self.settingsDialog.exec_()
        if result == self.settingsDialog.Accepted:
            self.mooseHandler.addSimPathList(self.settingsDialog.simPathList())

    def popupFieldMenu(self, clickpoint):
        print 'PopupFieldMenu'
        index = self.propertyEditor.indexAt(clickpoint)
        data = self.propertyModel.data(self.propertyModel.createIndex(index.row(), 0))
        print data
        menu = QtGui.QMenu(self.propertyEditor)
        self.actionPlotField = menu.addAction('Plot this field')
        self.connect(self.actionPlotField, QtCore.SIGNAL('triggered()'), self.plotThisFieldSlot)
        menu.popup(clickpoint)

    def plotThisFieldSlot(self):
        print 'plotThisFieldSlot'
        moose_object = self.modelTreeWidget.currentItem().getMooseObject()
        row = self.propertyEditor.currentIndex().row()
        index = self.propertyModel.createIndex(row, 0)
        print index.row(), index.column()
        field_name = self.propertyModel.data(index)        
        table = self.mooseHandler.createTableForMolecule(moose_object, field_name)
        print table.path
        
# 
# mainwin.py ends here
