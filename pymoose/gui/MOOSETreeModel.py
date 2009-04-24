# MOOSETreeModel.py --- 
# 
# Filename: MOOSETreeModel.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Thu Apr 16 14:37:07 2009 (+0530)
# Version: 
# Last-Updated: Thu Apr 16 21:41:16 2009 (+0530)
#           By: subhasis ray
#     Update #: 119
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
from PyQt4 import QtCore, QtGui

sys.path.append("/home/subha/src/moose/pymoose")

import moose

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
        return 1

    def data(self, column=0):
        return self.itemData

    def parent(self):
        return self.parentItem

    def row(self):
        if self.parentItem:
            return self.parentItem.childItems.index(self)

    def insertChildren(self, position,  count=1, columns=1):
        if position < 0 or position > len(self.childItems):
            return False;

        for row in range(count):
            data = QtCore.QVariant()
            item = TreeItem(data, self)
            self.childItems.insert(position, item);
        return True;

    def setData(self, data):
        self.itemdata = data

class MOOSETreeModel(QtCore.QAbstractItemModel):
    rootData = [moose.Neutral("/")]

    def __init__(self, data, parent=None):
        QtCore.QAbstractItemModel.__init__(self, parent)
        self.rootItem = TreeItem(MOOSETreeModel.rootData)
        self.itemdata = data

    def columnCount(self, parent):
        return 1

    def data(self, index, role):
        if not index.isValid() or role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()
        item = index.internalPointer()
        
        return QtCore.QVariant(item.data().name)

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def index(self, row, column, parent):
        if row < 0 or column != 0 or row >= self.rowCount(parent):
            return QtCore.QModelIndex()
        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()
        childItem = parentItem.child(row)
        if childItem:
            return self.createIndex(row, column, childItem)
        else:
            return QtCore.QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QtCore.QModelIndex()
        childItem = index.internalPointer()
        parentItem = childItem.parent()
        if parentItem == self.rootItem:
            return QtCore.QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

    def rowCount(self, parent):
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()

        return parentItem.childCount()

    def getItem(self, index):
        if index and index.isValid():
            item = index.internalPointer()
            if item:
                return item
        return self.rootItem

    def insertColumn(self, position, columns, parent):
        return False
    
    def insertRows(self, position, rows, parent):
        parentItem = self.getItem(parent)
        self.beginInsertRows(parent, position, position) # Only one object at a time
        parentItem.insertChildren(position, rows)
        self.endInsertRows()
        return True

    def setData(self, index, value, role):
        if role != QtCore.Qt.EditRole:
            return False
        item = self.getItem(index)
        item.setData(value)
        return True


if __name__ == "__main__":
    c = moose.Compartment("c")
    d = moose.HHChannel("chan", c)
    tc = TreeItem(c)
    td = TreeItem(d, tc)
    app = QtGui.QApplication(sys.argv)
    model = MOOSETreeModel(c)
    view = QtGui.QTreeView()
    view.setModel(model)
    model.insertRows(0, 1,view.selectionModel().currentIndex())
    model.setData(model.index(0, 0, tc), d, QtCore.Qt.EditRole)
    mainW = QtGui.QMainWIndow()
    mainW.setCentralWidget(view)
    main.show()
    sys.exit(app.exec_())
                  
# 
# MOOSETreeModel.py ends here
