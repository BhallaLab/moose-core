# MOOSETreeModel.py --- 
# 
# Filename: MOOSETreeModel.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Thu Apr 16 14:37:07 2009 (+0530)
# Version: 
# Last-Updated: Tue Jun 23 15:15:13 2009 (+0530)
#           By: subhasis ray
#     Update #: 255
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
        ret =  len(self.childItems)
        print 'childCount():', ret
        return ret

    def columnCount(self):
        return 1

    def data(self, column=0):
#         print 'data(): returning ', self.itemData
        return self.itemData


    def parent(self):
        return self.parentItem

    def row(self):
        if self.parentItem:
            ret = self.parentItem.childItems.index(self)
            print 'row():', ret
            return ret

#     def insertChildren(self, position,  count=1, columns=1):
#         if position < 0 or position > len(self.childItems):
#             return False;

#         for row in range(count):
#             print 'Insert children: adding empty QVariant'
#             data = QtCore.QVariant()
#             item = TreeItem(data, self)
#             self.childItems.insert(position, item);
#         return True;

    def setData(self, data):
        self.itemdata = data


class MOOSETreeModel(QtCore.QAbstractItemModel):
    rootData = moose.Neutral("/")

    def __init__(self, data, parent=None):
        QtCore.QAbstractItemModel.__init__(self, parent)
        self.rootItem = TreeItem(MOOSETreeModel.rootData)
        self.itemdata = data

    def columnCount(self, parent):
        print 'columnCount():',
        ret = 1
        if parent.isValid():
            ret = parent.internalPointer().columnCount()
            print 'parent valid'
        else:
            print 'rootitems', parent
            ret = self.rootItem.columnCount()
        print ret
        return ret

    def data(self, index, role):
        print 'data():'
        if not index.isValid():
            print 'Indexd is not valid'
            return QtCore.QVariant()
        item = index.internalPointer()
        item_data = item.data()
        if isinstance(item_data, moose.PyMooseBase):
            if role == QtCore.Qt.DisplayRole:
                print 'DisplayRole:', item_data.name
                return QtCore.QVariant(QtCore.QString(item_data.name))
            elif role == QtCore.Qt.ToolTipRole:
                return QtCore.QVariant(QtCore.QString(item_data.className))
            else:
                print 'role is:', role,
                return QtCore.QVariant(item_data.path)
                
        elif isinstance(item_data, QtCore.QVariant):
            print 'This is a QVariant'
            return item_data
        else:
            print 'Nonr of the known types'
            return QtCore.QVariant()

    def flags(self, index):
        print 'flags():'
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def index(self, row, column, parent=QtCore.QModelIndex()):
        print 'index():'
        if row < 0 or column < 0 or row >= self.rowCount(parent) or column >= self.columnCount(parent):
            print 'index - invalid'
            return QtCore.QModelIndex()
        if not parent.isValid():
            print 'parent - invalid, using root'
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()
        childItem = parentItem.child(row)
        if childItem:
            print 'child item', childItem
            return self.createIndex(row, column, childItem)
        else:
            print 'no child item'
            return QtCore.QModelIndex()

    def parent(self, index):
        print 'parent():'
        if not index.isValid():
            print 'parent invalid'
            return QtCore.QModelIndex()
        childItem = index.internalPointer()
        parentItem = childItem.parent()
        if parentItem == self.rootItem:
            print 'root item'
            return QtCore.QModelIndex()
        return self.createIndex(parentItem.row(), 0, parentItem)

    def rowCount(self, parent):
        print 'rowCount():'
        ret = 0
        if not parent.isValid():
            print 'parent not valid', parent
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()
        ret = parentItem.childCount()
        print ret
        return ret

    def getItem(self, index):
        print 'getItem():'
        if index and index.isValid():
            item = index.internalPointer()
            if item:
                return item
        return self.rootItem

    def insertColumn(self, position, columns, parent):
        print 'insertColumn:'
        return False

    def hasChildren(self, parent):
        if self.rowCount(parent) > 0:
            return True
        return False

#     def insertRows(self, position, rows, parent):
#         parentItem = self.getItem(parent)
#         self.beginInsertRows(parent, position, position) # Only one object at a time
#         parentItem.insertChildren(position, rows)
#         self.endInsertRows()
#         return True

    def setData(self, index, value, role):
        if role != QtCore.Qt.EditRole:
            return False
        item = self.getItem(index)
        item.setData(value)
        return True

    def setupModelData(self, root, parent=None):
        if not isinstance(root, TreeItem):
            print 'Error: must pass a TreeItem as root'
            return
        parents = [root]
        while parents:
            current = parents.pop()
            obj = current.data()
            if not isinstance(obj, moose.PyMooseBase):
                print 'ERROR: not  a moose object'
            else:
                print 'Object:', obj.path
                for child in obj.children():
                    childData = moose.Neutral(child)
                    print 'child:', childData.path
                    childItem = TreeItem(childData, current)
                    current.appendChild(childItem)
                    parents.append(childItem)

if __name__ == "__main__":
    c = moose.Compartment("c")
    d = moose.HHChannel("chan", c)
#     tc = TreeItem(c)
#     td = TreeItem(d, tc)
    rootItem = TreeItem(MOOSETreeModel.rootData)
    app = QtGui.QApplication(sys.argv)
    model = MOOSETreeModel(MOOSETreeModel.rootData)
    model.setupModelData(rootItem)
    view = QtGui.QTreeView()
    view.setModel(model)
    model.insertRows(0, 1, view.selectionModel().currentIndex())
#     model.setData(model.index(0, 0, tc), d, QtCore.Qt.EditRole)
    mainW = QtGui.QMainWindow()
    mainW.setCentralWidget(view)
    mainW.show()

    sys.exit(app.exec_())
                  
# 
# MOOSETreeModel.py ends here
