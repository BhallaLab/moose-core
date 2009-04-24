# MOOSEMain.py --- 
# 
# Filename: MOOSEMain.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Sun Apr 12 18:52:42 2009 (+0530)
# Version: 
# Last-Updated: Fri Apr 17 10:00:08 2009 (+0530)
#           By: subhasis ray
#     Update #: 43
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

from PyQt4 import QtGui
from PyQt4.Qt import Qt
from PyQt4 import QtCore

from MOOSEToolBox import *
from MOOSEWorkspace import *

class MOOSEMain(QtGui.QMainWindow):
    def __init__(self, *args):
        QtGui.QMainWindow.__init__(self, *args)
        self.toolDock = QtGui.QDockWidget(self.tr("MOOSE Classes"), self)
        self.toolDock.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.toolBox = MOOSEToolBox(self.toolDock)
        self.toolDock.setWidget(self.toolBox)
        self.workspace = MOOSEWorkspace(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.toolDock)
        self.objectEditDock = QtGui.QDockWidget(self.tr("Object properties"), self)
        self.objectEditDock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.objectEditor = QtGui.QTableView(self.objectEditDock)
        self.objectEditDock.setWidget(self.objectEditor)
        self.addDockWidget(Qt.RightDockWidgetArea, self.objectEditDock)
        self.setCentralWidget(self.workspace)
        for widget in self.toolBox.listWidgets:
            self.connect(widget, QtCore.SIGNAL("itemClicked(QListWidgetItem*)"), self.workspace.addElementSlot)
            
import sys
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    main = MOOSEMain()
    main.show()
    sys.exit(app.exec_())


# 
# MOOSEMain.py ends here
