# moosetree.py --- 
# 
# Filename: moosetree.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Tue Jun 23 18:54:14 2009 (+0530)
# Version: 
# Last-Updated: Sun Jul  5 01:35:11 2009 (+0530)
#           By: subhasis ray
#     Update #: 137
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

import moose
import sys
from PyQt4 import QtCore, QtGui

class MooseTreeItem(QtGui.QTreeWidgetItem):
    def __init__(self, *args):
	QtGui.QTreeWidgetItem.__init__(self, *args)
	self.mooseObj_ = None
    
    def setMooseObject(self, mooseObject):
	if isinstance(mooseObject, moose.Id):
	    self.mooseObj_ = moose.Neutral(mooseObject)
	elif isinstance(mooseObject, moose.PyMooseBase):
	    self.mooseObj_ = mooseObject
	else:
	    raise Error
	self.setText(0, QtCore.QString(self.mooseObj_.name))
	self.setToolTip(0, QtCore.QString('class:' + self.mooseObj_.className))

    def getMooseObject(self):
	return self.mooseObj_

    def updateSlot(self, text):
	self.setText(0, QtCore.QString(self.mooseObj_.name))

class MooseTreeWidget(QtGui.QTreeWidget):
    def __init__(self, *args):
	QtGui.QTreeWidget.__init__(self, *args)
	self.rootObject = moose.Neutral('/')
	self.itemList = []
	self.setupTree(self.rootObject, self, self.itemList)
        self.setCurrentItem(self.itemList[0]) # Make root the default item

    def setupTree(self, mooseObject, parent, itemlist):
	item = MooseTreeItem(parent)
	item.setMooseObject(mooseObject)
	itemlist.append(item)
	for child in mooseObject.children():
	    childObj = moose.Neutral(child)
	    self.setupTree(childObj, item, itemlist)

	return item

    def recreateTree(self):
        self.clear()
        self.itemList = []
        self.setupTree(moose.Neutral('/'), self, self.itemList)

    def insertMooseObjectSlot(self, class_name):
        try:
            class_name = str(class_name)
            class_obj = eval('moose.' + class_name)
            current = self.currentItem()
            new_item = MooseTreeItem(current)
            parent = current.getMooseObject()
#             print 'creating new', class_name, 'under', parent.path
            new_obj = class_obj(class_name, parent)
            new_item.setMooseObject(new_obj)
            current.addChild(new_item)
            self.itemList.append(new_item)
        except AttributeError:
            print class_name, ': no such class in module moose'

if __name__ == '__main__':
    c = moose.Compartment("c")
    d = moose.HHChannel("chan", c)
    app = QtGui.QApplication(sys.argv)
    widget = MooseTreeWidget()
#     widget = QtGui.QTreeWidget()
#     items = []
#     root = moose.Neutral('/')
#     parent = widget
#     item = setupTree(root, widget, items)
#     while stack:
# 	mooseObject = stack.pop()
# 	item = QtGui.QTreeWidgetItem(parent)
# 	item.setText(0, widget.tr(mooseObject.name))
# 	parent = item
# 	for child in mooseObject.children():
# 	    stack.append(moose.Neutral(child))
    widget.show()
    sys.exit(app.exec_())

# 
# moosetree.py ends here
