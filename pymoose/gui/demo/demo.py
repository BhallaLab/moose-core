# demo.py --- 
# 
# Filename: demo.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Fri Feb  5 23:19:44 2010 (+0530)
# Version: 
# Last-Updated: Sat Feb  6 00:05:48 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 41
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
import os

from PyQt4 import QtCore, QtGui
from PyQt4.Qt import Qt

class MooseDemoWidget(QtGui.QWidget):
    '''A common launcher for all the demos'''
    def __init__(self):
	QtGui.QWidget.__init__(self)
	self.setWindowTitle(self.tr('MOOSE DEMOS'))
	
	# This should be set to the location of the ${MOOSE}/DEMOS
	# directory. Can be chosen via the push button 'DEMOS
	# directory'
	self.demoDir = '../../../DEMOS'

	self.demoDirButton = QtGui.QPushButton(self.tr('DEMOS Directory'))
	self.demoDirLabel = QtGui.QLabel(self.tr(os.path.dirname(self.demoDir)), self)
	

	self.glCellButton = QtGui.QPushButton(self.tr('GL Cell'), self)
	self.glCellButton.setToolTip(self.tr('<html>Load the OpenGL based cell viewer with a sample mitral cell model.</html>'))

	self.glViewButton = QtGui.QPushButton(self.tr('GL View'), self)
	self.glViewButton.setToolTip(self.tr('<html>Load the OpenGL based object viewer with the same mitral cell</html>'))

	self.squidButton = QtGui.QPushButton(self.tr('Squid Axon'), self)
	self.squidButton.setToolTip(self.tr('<html>Load the Squid Axon demo.</html>'))
	layout = QtGui.QGridLayout(self)
	layout.addWidget(self.demoDirButton, 0, 0)
	layout.addWidget(self.demoDirLabel, 0, 1)
	layout.addWidget(self.glCellButton, 1, 0)
	layout.addWidget(self.glViewButton, 2, 0)
	layout.addWidget(self.squidButton, 3, 0)
	self.setLayout(layout)
	
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    mw = MooseDemoWidget()
    mw.show()
    app.exec_()


# 
# demo.py ends here
