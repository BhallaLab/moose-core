# MOOSEToolBox.py --- 
# 
# Filename: MOOSEToolbox.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Sun Apr 12 16:29:09 2009 (+0530)
# Version: 
# Last-Updated: Sun Apr 12 19:16:11 2009 (+0530)
#           By: subhasis ray
#     Update #: 10
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

from MOOSEClasses import *

import moose_rc

class MOOSEToolItem(QtGui.QListWidgetItem):
    """Items in MOOSE toolbox"""
    def __init__(self, item_name, category, parent):
        QtGui.QListWidgetItem.__init__(self, parent)
        self.setText(item_name)
        self.setIcon(QtGui.QIcon(":/icons/"+ category + "/" + item_name + ".ico"))

class MOOSEToolBox(QtGui.QToolBox):
    def __init__(self, *args):
        QtGui.QToolBox.__init__(self, *args)
        self.listWidgets = []
        for category in MOOSEClasses.categories:
            page = QtGui.QWidget(self)
            page.setObjectName(category + "Page")
            tool_list = QtGui.QListWidget(page)
            self.listWidgets.append(tool_list)
            for item_name in MOOSEClasses.class_dict[category]:
                tool_item = MOOSEToolItem(item_name, category, tool_list)
            self.addItem(page, category)


import sys
if __name__ == "__main__":
    app = QtGui.QApplication([])
    tb = MOOSEToolBox()
    tb.show()
    sys.exit(app.exec_())




# 
# MOOSEToolBox.py ends here
