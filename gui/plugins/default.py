# default.py --- 
# 
# Filename: default.py
# Description: 
# Author: 
# Maintainer: 
# Created: Tue Nov 13 15:58:31 2012 (+0530)
# Version: 
# Last-Updated: Thu Dec  6 17:43:05 2012 (+0530)
#           By: subha
#     Update #: 239
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# The default placeholder plugin for MOOSE
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
from PyQt4.Qt import Qt
import moose
from mplugin import MoosePluginBase, EditorBase, EditorWidgetBase

class MoosePlugin(MoosePluginBase):
    """Default plugin for MOOSE GUI"""
    def __init__(self, root, mainwindow):
        MoosePluginBase.__init__(self, root, mainwindow)

    def getPreviousPlugin(self):
        return None

    def getNextPlugin(self):
        return None

    def getAdjacentPlugins(self):
        return []

    def getViews(self):
        return self._views

    def getCurrentView(self):
        return self.currentView

    def getEditorView(self):
        if not hasattr(self, 'editorView'):
            self.editorView = MooseEditorView(self)
            self.currentView = self.editorView
        return self.editorView


class MooseEditorView(EditorBase):
    """Default editor.

    TODO: Implementation - display moose element tree as a tree or as
    boxes inside boxes

    """
    def __init__(self, plugin):
        EditorBase.__init__(self, plugin)

    def getToolPanes(self):
        return super(MooseEditorView, self).getToolPanes()

    def getLibraryPane(self):
        return super(MooseEditorView, self).getLibraryPane()

    def getOperationsWidget(self):
        return super(MooseEditorView, self).getOperationsPane()

    def getCentralWidget(self):
        """Retrieve or initialize the central widget.

        Note that we call the widget's setModelRoot() function
        explicitly with the plugin's modelRoot as the argument. This
        enforces an update of the widget display with the current
        modelRoot.

        This function should be overridden by any derived class as it
        has the editor widget class hard coded into it.

        """
        if self._centralWidget is None:
            self._centralWidget = DefaultEditorWidget()
            self._centralWidget.setModelRoot(self.plugin.modelRoot)
        return self._centralWidget

class DefaultEditorWidget(EditorWidgetBase):
    """Editor widget for default plugin. 
    
    Currently does nothing. Plugin-writers should code there own
    editor widgets derived from EditorWidgetBase.
    """
    def __init__(self, *args):
        EditorWidgetBase.__init__(self, *args)
        # self.qmodel = MooseTreeModel(self)
        # self.qview = QtGui.QTreeView(self)
        # self.qview.setModel(self.qmodel)
    
    def updateModelView(self):
        # TODO: implement a tree / box widget
        #print 'updateModelView', self.modelRoot
        # self.qmodel.setupModelData(moose.ematrix(self.modelRoot))
        pass


class MooseTreeModel(QtCore.QAbstractItemModel):
    """Tree model for the MOOSE element tree.
    
    This is not going to work as the MOOSE tree nodes are
    inhomogeneous. The parent of a node is an melement, but the
    children of an melement are ematrix objects.

    Qt can handle only homogeneous tere nodes.
    """
    def __init__(self, *args):
        super(MooseTreeModel, self).__init__(*args)
        self.rootItem = moose.element('/')

    def setupModelData(self, root):
        self.rootItem = root        
        print 'setupModelData', self.rootItem
    
    def index(self, row, column, parent):
        print 'index', row, column, parent.internalPointer().path
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()
        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()
        childItem = parentItem.children[row]
        if childItem.path == '/':
            return QtCore.QModelIndex()
        return self.createIndex(row, column, childItem)

    def parent(self, index):
        if not index.isValid():
            return QtCore.QModelIndex()
        childItem = index.internalPointer()
        print 'parent():', childItem.path
        parentItem = childItem.parent()
        if parentItem == self.rootItem:
            return QtCore.QModelIndex()
        return self.createIndex(parentItem.parent.children.index(parentItem), parentItem.getDataIndex(), parentItem)

    def rowCount(self, parent):
        print 'Row count', parent
        if not parent.isValid():
            parentItem = self.rootItem
        else:
            parentItem = parent.internalPointer()
        ret = len(parentItem.children)
        print 'rowCount()', ret
        return ret

    def columnCount(self, parent):
        print 'Column count', parent,
        if parent.isValid():
            print '\t',parent.internalPointer().path
            return len(parent.internalPointer())
        else:
            print '\tInvalid parent',
        ret = len(self.rootItem)
        print '\t', ret
        return ret

    def data(self, index, role):
        print 'data', index
        if not index.isValid():
            return None
        item = index.internalPointer()
        print '\t', item.name, role
        return QtCore.QVariant(item[index.column()].name)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:  
            return QtCore.QVariant('Model Tree')
        return None

    def flags(self, index):
         if not index.isValid():
             return QtCore.Qt.NoItemFlags
         return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        
    
class MooseTreeItem(QtGui.QTreeWidgetItem):
    def __init__(self, *args):
	QtGui.QTreeWidgetItem.__init__(self, *args)
	self.mobj = None

    def setObject(self, element):
        if isinstance(element, moose.marray):
            self.mobj = marray[0]
        elif isinstance(element, moose.melement):
	    self.mobj = element
	elif isinstance(element, str):
	    self.mobj = moose.element(element)
	else:
            raise TypeError('Takes a path or an element or an array')
	self.setText(0, QtCore.QString(self.mobj.name))
	self.setText(1, QtCore.QString(self.mobj.class_))
	#self.setToolTip(0, QtCore.QString('class:' + self.mooseObj_.className))

    def updateSlot(self):
	self.setText(0, QtCore.QString(self.mobj.name))


class MooseTreeWidget(QtGui.QTreeWidget):
    """Widget for displaying MOOSE model tree.

    """
    # Author: subhasis ray
    #
    # Created: Tue Jun 23 18:54:14 2009 (+0530)
    #
    # Updated for moose 2 and multiscale GUI: 2012-12-06


    def __init__(self, *args):
	QtGui.QTreeWidget.__init__(self, *args)
        self.header().hide()
	self.rootObject = '/'
	self.itemList = []
	# self.setupTree(self.rootObject, self, self.itemList)
        # self.setCurrentItem(self.itemList[0]) # Make root the default item
        # self.setColumnCount(2)	
	# self.setHeaderLabels(['Moose Object                    ','Class']) 	#space as a hack to set a minimum 1st column width
	# self.header().setResizeMode(QtGui.QHeaderView.ResizeToContents)
	# self.expandToDepth(0)
        # self.mooseHandler = None

    def setupTree(self, obj, parent, itemlist):
	item = MooseTreeItem(parent)
	item.setObject(obj)
	itemlist.append(item)
	for childarray in obj.children:
            for child in childarray:
                self.setupTree(child, item, itemlist)
        
	return item

    def recreateTree(self):
        self.clear()
        self.itemList = []
        self.setupTree(moose.Neutral('/'), self, self.itemList)
        if self.mooseHandler:
            self.setCurrentItem(self.mooseHandler.getCurrentElement())
        self.expandToDepth(0)

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

# 
# default.py ends here
