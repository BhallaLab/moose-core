# moosetree.py --- 
# 
# Filename: moosetree.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Tue Jun 23 18:54:14 2009 (+0530)
# Version: 
# Last-Updated: Fri Feb 25 11:25:01 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 203
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

import moose

import config

def pathMatches(oldpath, newname):
    """Check if replacing the object name with newname in oldpath
    corresponds to a valid object."""
    old_parent_path_end = oldpath.rfind('/')
    old_parent_path = oldpath[:old_parent_path_end]
    newpath = old_parent_path + '/' + newname
    print 'matchesPath', newpath
    return config.context.exists(newpath)
    

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

    def updateSlot(self):
	self.setText(0, QtCore.QString(self.mooseObj_.name))

class MooseTreeWidget(QtGui.QTreeWidget):
    def __init__(self, *args):
	QtGui.QTreeWidget.__init__(self, *args)
        self.header().hide()
	self.rootObject = moose.Neutral('/')
	self.itemList = []
	self.setupTree(self.rootObject, self, self.itemList)
        self.setCurrentItem(self.itemList[0]) # Make root the default item
        self.mooseHandler = None

    def setMooseHandler(self, handler):
        self.mooseHandler = handler

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
        if self.mooseHandler:
            self.setCurrentItem(self.mooseHandler.getCurrentElement())

    def insertMooseObjectSlot(self, class_name):
        """Creates an instance of the class class_name and inserts it
        under currently selected element in the model tree."""
        try:
            class_name = str(class_name)
            class_obj = eval('moose.' + class_name)
            current = self.currentItem()
            new_item = MooseTreeItem(current)
            parent = current.getMooseObject()
            new_obj = class_obj(class_name, parent)
            new_item.setMooseObject(new_obj)
            current.addChild(new_item)
            self.itemList.append(new_item)
            self.emit(QtCore.SIGNAL('mooseObjectInserted(PyQt_PyObject)'), new_obj)
        except AttributeError:
            config.LOGGER.error('%s: no such class in module moose' % (class_name))

    def setCurrentItem(self, item):
        if isinstance(item, QtGui.QTreeWidgetItem):
            QtGui.QTreeWidget.setCurrentItem(self, item)
        elif isinstance(item, moose.PyMooseBase):
            for entry in self.itemList:
                if entry.getMooseObject().path == item.path:
                    QtGui.QTreeWidget.setCurrentItem(self, entry)
        elif isinstance(item, str):
            for entry in self.itemList:
                if entry.getMooseObject().path == item:
                    QtGui.QTreeWidget.setCurrentItem(self, entry)
        else:
            raise Exception('Expected QTreeWidgetItem/moose object/string. Got: %s' % (type(item)))


    def updateItemSlot(self, mooseObject):
        for changedItem in (item for item in self.itemList if mooseObject.id == item.mooseObj_.id):
            break
        changedItem.updateSlot()
        
    def pathToTreeChild(self,moosePath):	#traverses the tree, itemlist already in a sorted way 
    	path = str(moosePath)
    	for item in self.itemList:
    		if path==item.mooseObj_.path:
    			return item

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
