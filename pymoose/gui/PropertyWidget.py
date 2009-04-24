# PropertyWidget.py --- 
# 
# Filename: PropertyWidget.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Sun Apr 19 00:14:53 2009 (+0530)
# Version: 
# Last-Updated: Sun Apr 19 00:22:34 2009 (+0530)
#           By: subhasis ray
#     Update #: 11
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

from PyQt4 import QtGui, QtCore

from MOOSEPropertyModel import PropertyModel


class PropertyWidget(QtGui.QTableView):
    def __init__(self, *args):
	QtGui.QTableView.__init__(self, *args)

#     def setObject(self, mooseObject):
# 	model = PropertyModel(mooseObject)
# 	self.setModel(model)

import sys
sys.path.append("/home/subha/src/moose/pymoose")
import moose

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    c = moose.HHChannel("chan")
    p = PropertyWidget()
    p.setObject(c)
    p.show()
# 
# PropertyWidget.py ends here
