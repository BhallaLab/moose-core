# mainwin.py --- 
# 
# Filename: mainwin.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Tue Jun 16 11:38:46 2009 (+0530)
# Version: 
# Last-Updated: Thu Jun 18 00:10:35 2009 (+0530)
#           By: subhasis ray
#     Update #: 311
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
import math
from PyQt4.Qt import Qt
from PyQt4 import QtCore, QtGui
from PyQt4 import Qwt5 as Qwt
import PyQt4.Qwt5.qplt as qplt
import PyQt4.Qwt5.anynumpy as numpy


from ui_mainwindow import Ui_MainWindow
from moosehandler import MHandler

class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    """Main Window for MOOSE GUI"""
    def __init__(self, loadFile=None, fileType=None):
	QtGui.QMainWindow.__init__(self)
	self.setupUi(self)
        self.isModelLoaded = False
	self.mooseHandler = MHandler()
        self.runTimeLineEdit.setText(self.tr(str(self.mooseHandler.runTime)))
	self.currentDirectory = '.'
	self.fileType = fileType
	self.fileTypes = MHandler.file_types
	self.dataPlotMap = {} # Contains a map of MOOSE tables to QwtPlot objects
	self.filter = ''
	self.setupActions()
        if loadFile is not None:
            if fileType is None:
                errorDialog = QtGui.QErrorMessage.showMessage('<p>You must specify a model type as the final argument to this program</p>')
            else:
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

	self.connect(self.actionAbout_MOOSE,
		     QtCore.SIGNAL('triggered()'),
		     self.showAbout_MOOSE)

	self.connect(self.actionReset,
		     QtCore.SIGNAL('triggered()'),
		     self.reset)

	self.connect(self.actionStart,
		     QtCore.SIGNAL('triggered()'),
		     self.run)


    def loadFileDialog(self):
	fileDialog = QtGui.QFileDialog(self)
	fileDialog.setFileMode(QtGui.QFileDialog.ExistingFile)
	ffilter = ''
	for key in self.fileTypes.keys():
	    ffilter = ffilter + key + ';;'
	ffilter = ffilter[:-2]
	fileDialog.setFilter(self.tr(ffilter))
	
	if fileDialog.exec_():
	    fileNames = fileDialog.selectedFiles()
	    fileFilter = fileDialog.selectedFilter()
	    fileType = self.fileTypes[str(fileFilter)]
	    print 'file type:', fileType
	    directory = fileDialog.directory() # Potential bug: if user types the whole file path, does it work? - no but gives error message
	    for fileName in fileNames: 
		print fileName
		self.load(fileName, fileType)


    def loadIzhikevich_Neurons_Tutorial(self):
	self.mooseHandler.load('../examples/izhikevich/Izhikevich.py', 'MOOSE')


    def loadSquid_Axon_Tutorial(self):
	self.mooseHandler.load('../examples/squid/qtSquid.py', 'MOOSE')


    def showAbout_MOOSE(self):
	about = QtCore.QT_TR_NOOP('<p>MOOSE is the Multi-scale Object Oriented Simulation Environment.</p>'
				  '<p>It is a general purpose simulation environment for computational neuroscience and chemical kinetics.</p>'
				  '<p>Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS</p>'
				  '<p>It is made available under the terms of the GNU Lesser General Public License version 2.1. See the file COPYING.LIB for the full notice.</p>')
	aboutDialog = QtGui.QMessageBox.information(self, self.tr('About MOOSE'), about)

	
    def run(self):
        try:
            time = float(str(self.runTimeLineEdit.text()))
        except ValueError:
            print 'Error in converting runtime to float'
            time = 1e-2
            self.runTimeLineEdit.setText(self.tr(time))

        print 'Going to run'
        if not self.isModelLoaded:
            QtGui.QErrorMessage.showMessage('<p>You must load a model first.</p>')
            return
        self.mooseHandler.run(time)
        for dataTable, dataPlot in self.dataPlotMap.items():
            ydata = numpy.array(dataTable)
            curve = Qwt.QwtPlotCurve(self.tr(dataTable.name))
            xdata = numpy.linspace(0, time, len(dataTable))
            curve.setData(xdata, ydata)
            curve.setPen(QtCore.Qt.red)
            curve.attach(dataPlot)
            dataPlot.replot()

    def reset(self):
        if not self.isModelLoaded:
            QtGui.QErrorMessage.showMessage('<p>You must load a model first.</p>')
        else:
            self.mooseHandler.reset()

    def load(self, fileName, fileType):
        if self.isModelLoaded:
            import subprocess
            subprocess.call(['python', 'main.py', fileName, fileType])
        self.mooseHandler.load(fileName, fileType)
        dataTables = self.mooseHandler.getDataObjects()
        rows = math.ceil(math.sqrt(len(dataTables)))
        cols = math.ceil(float(len(dataTables)) / rows)
        row = 0
        col = 0
        self.plotsGridLayout = QtGui.QGridLayout(self.plotsTab)
        self.plotsTab.setLayout(self.plotsGridLayout)
        expansion = self.tabWidget.sizePolicy().expandingDirections()
        for table in dataTables:
            plot = Qwt.QwtPlot(self.plotsTab)
            self.dataPlotMap[table] = plot
            self.plotsGridLayout.addWidget(plot, row, col)
            if col >= cols:
                row += 1
                col = 0
            else:
                col += 1
        self.isModelLoaded = True
        
# 
# mainwin.py ends here
