# mainwin.py --- 
# 
# Filename: mainwin.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Tue Jun 16 11:38:46 2009 (+0530)
# Version: 
# Last-Updated: Wed Jun 17 01:41:49 2009 (+0530)
#           By: subhasis ray
#     Update #: 157
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

from PyQt4.Qt import Qt
from PyQt4 import QtCore, QtGui

from ui_mainwindow import Ui_MainWindow
from moosehandler import MHandler

class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    """Main Window for MOOSE GUI"""
    def __init__(self, *args):
	QtGui.QMainWindow.__init__(self, *args)
	self.mooseHandler = MHandler()
	self.currentDirectory = '.'
	self.fileType = None
	self.fileTypes = MHandler.file_types
	
	self.filter = ''
	self.setupUi(self)
	self.setupActions()

    def setupActions(self):
	self.connect(self.actionQuit,
		     QtCore.SIGNAL('triggered()'), 
		     QtGui.qApp, 
		     QtCore.SLOT('closeAllWindows()'))
		     
	self.connect(self.actionLoad, 
		     QtCore.SIGNAL('triggered()'),		   
		     self.load)

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
		     self.mooseHandler.reset)

	self.connect(self.actionStart,
		     QtCore.SIGNAL('triggered()'),
		     self.run)

    def load(self):
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
	    print 'file type:', self.fileType
	    directory = fileDialog.directory() # Potential bug: if user types the whole file path, does it work? - no but gives error message
	    for fileName in fileNames: 
		print fileName
		self.mooseHandler.load(fileName, fileType)

    def loadIzhikevich_Neurons_Tutorial(self):
	self.mooseHandler.load('../examples/izhikevich/Izhikevich.py', 'MOOSE')

    def loadSquid_Axon_Tutorial(self):
	self.mooseHandler.load('../examples/squid/qtSquid.py', 'MOOSE')

    def showAbout_MOOSE(self):
	about = QtCore.QT_TR_NOOP('<p>MOOSE is the Multi-scale Object Oriented Simulation Environment.</p>'
				  '<p>It is a general purpose simulation environment for computational neuroscience and chemical kinetics.</p>'
				  '<p>Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS</p>'
				  '<p>It is made available under the terms of the GNU Lesser General Public License version 2.1. See the file COPYING.LIB for the full notice.</p>')
	aboutDialog = QtGui.QMessageBox.information(self, self.tr('About MOOSE'), about)

	
    def run(self):
	time = 1.0 # Test code - will come from a text box
	self.mooseHandler.run(time)
# 
# mainwin.py ends here
