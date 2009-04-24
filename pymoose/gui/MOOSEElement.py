# MOOSEElement.py --- 
# 
# Filename: MOOSEElement.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Sun Apr 12 17:16:10 2009 (+0530)
# Version: 
# Last-Updated: Thu Apr 16 15:05:41 2009 (+0530)
#           By: subhasis ray
#     Update #: 98
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

import sys
sys.path.append("/home/subha/src/moose/pymoose")
import moose

object_dict = {}

class TreeItem:
    """Borrowed from PyQt examples"""
    def __init__(self, data, parent=None):
        self.parentItem = parent
        self.itemData = data
        self.childItems = []

    def appendChild(self, item):
        self.childItems.append(item)

    def child(self, row):
        return self.childItems[row]

    def childCount(self):
        return len(self.childItems)

    def columnCount(self):
        return len(self.itemData)

    def data(self, column):
        return self.itemData[column]

    def parent(self):
        return self.parentItem

    def row(self):
        if self.parentItem:
            return self.parentItem.childItems.index(self)

        return 0

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    m = MOOSEElement('a')
    m.show()
    sys.exit(app.exec_())
# 
# MOOSEElement.py ends here
