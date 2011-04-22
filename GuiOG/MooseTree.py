import moose
import sys
from PyQt4 import QtCore, QtGui

class MooseTreeItem(QtGui.QTreeWidgetItem): 	#from subhasis's code. moosetree.py
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
	self.setText(1, QtCore.QString(self.mooseObj_.className))	
	#self.setToolTip(0, QtCore.QString('class:' + self.mooseObj_.className))

    def getMooseObject(self):
	return self.mooseObj_

    def updateSlot(self):
	self.setText(0, QtCore.QString(self.mooseObj_.name))


class MooseTreeWidget(QtGui.QTreeWidget):
    def __init__(self, *args):
	QtGui.QTreeWidget.__init__(self, *args)
	self.rootObject = moose.Neutral('/')
	self.itemList = []
	self.setupTree(self.rootObject, self, self.itemList)
        self.setCurrentItem(self.itemList[0]) 					# Make root the default item
	self.setColumnCount(2)	
	self.setHeaderLabels(['Moose Object                    ','Class']) 	#space as a hack to set a minimum 1st column width
	self.header().setResizeMode(QtGui.QHeaderView.ResizeToContents)
	self.expandToDepth(0)
	self.setWindowTitle('Moose Tree')

    def setupTree(self, mooseObject, parent, itemlist):
	item = MooseTreeItem(parent)
	item.setMooseObject(mooseObject)
	itemlist.append(item)
	for child in mooseObject.children():
	    childObj = moose.Neutral(child)
	    self.setupTree(childObj, item, itemlist)
	return item
    
    def mooseTreeItemClick(self, item, column):
	obj = item.getMooseObject()
	#print item, obj.path	

    def recreateTree(self):
        self.clear()
        self.itemList = []
        self.setupTree(moose.Neutral('/'), self, self.itemList)
	self.expandToDepth(0)
	self.show()

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
	    print 'Error'+class_name+ 'no such class in module moose'	            
		#config.LOGGER.error('%s: no such class in module moose' % (className))

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
    #widget = MooseTreeWidget()
    #widget.show()
    #sys.exit(app.exec_())

