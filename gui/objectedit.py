# objectedit.py --- 
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# This code is for a widget to edit MOOSE objects. We can now track if
# a field is a Value field and make it editable accordingly. There
# seems to be no clean way of determining whether the field is worth
# plotting (without a knowledge of the model/biology there is no way
# we can tell this). But we can of course check if the field is a
# numeric one.
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

import types
import sys

from PyQt4.Qt import Qt
from PyQt4 import QtCore
from PyQt4 import QtGui

# Local modules and moose.
import moose
import config

#app = QtGui.QApplication([])

class ObjectFieldsModel(QtCore.QAbstractTableModel):
    """Model the fields list for MOOSE objects.
    
    extra_fields -- list of fields that are of no use in the fields
                    editor.

    """
    extra_fields = ['this','me','parent','path','class','children','linearSize','objectDimensions','lastDimension','localNumField','pathIndices','msgOut','msgIn','diffConst','speciesId','Coordinates','neighbors','DiffusionArea','DiffusionScaling','x','x0','x1','dx','nx','y','y0','y1','dy','ny','z','z0','z1','dz','nz','coords','isToroid','preserveNumEntries']
    py_moose_fieldname_map = {}	
                              
    def __init__(self, mooseObject, parent=None):
        """Set up the model. 

        The table model has one moose field in each row.  A field that
        has a set method is editable. Fields listed in extra_fields
        are not shown.

        Members:

        fields -- list of the names of the fields in the object.

        plotNames -- lists the names of the available plot
                     windows. These are displayed as the targets in
                     the plot submenu / combobox.

        fieldFlags -- flags for each field. We calculate these ahead
                      of time by checking if the field can be set, if
                      it is a numerical (so can be dragged to a plot
                      window).

        """
        QtCore.QAbstractTableModel.__init__(self, parent)
        self.mooseObject = mooseObject
        self._header = ('Field', 'Value')
        self.fields = []
        self.plotNames = ['None']
        self.fieldFlags = {}
        self.fieldPlotNameMap = {}
        try:
            self.mooseObject = moose.element(mooseObject.getId())

        except AttributeError:
            config.LOGGER.error('Could not wrap object %s into class %s' % (mooseObject.path, className))
            return
        
        for fieldName in self.mooseObject.getFieldNames('valueFinfo'):
			#print fieldName
			if(fieldName in ObjectFieldsModel.extra_fields):
				continue
			else:
			    value = self.mooseObject.getField(fieldName)
			    #print value
			flag = Qt.ItemIsEnabled | Qt.ItemIsSelectable
			srchField = 'set_'+fieldName
			try:	
			    for fn in (fn for fn in moose.getFieldDict(self.mooseObject.class_,'destFinfo').keys() if fn.startswith(srchField)):
				    flag = flag | Qt.ItemIsEditable
			    value = mooseObject.getField(fieldName)
				
			except Exception, e:
				config.LOGGER.error("%s" % (e))

			self.fieldFlags[fieldName] = flag	
			self.fields.append(fieldName)            	
        self.insertRows(0, len(self.fields))
        
    def setData(self, index, value, role=Qt.EditRole):
        """Set field value or set plot flag.

        If a user tries to put an invalid value then the field is
        reset to default value. Restored to previous edit.
        """
        oldValue = str(index.data().toString())
        if not index.isValid() and index.row () >= len(self.fields):
            return False
        ret = True
        value = (value.toString()) # convert Qt datastructure to
                                      # Python datastructure
        #add_chait
        if value =='':
            value = oldValue
        field = self.fields[index.row()]
        
        if index.column() == 0: # This is the fieldname
            ret = False
        elif index.column() == 1: # This is the value column
            try:
                field = ObjectFieldsModel.py_moose_fieldname_map[field]
            except KeyError:
                pass
            if field == 'name':
                self.mooseObject.setField(field,str(value))
                self.emit(QtCore.SIGNAL('objectNameChanged(PyQt_PyObject)'), self.mooseObject)
            else:    
                try: 
                    self.mooseObject.setField(field,float(value))
                except ValueError: #folks entering text instead of numerals here!
                    print "Numeric value should be entered";
                    pass
            
        elif index.column() == 2 and role ==Qt.EditRole:
            try:
                self.fieldPlotNameMap[self.fields[index.row()]] = str(value)
                self.emit(QtCore.SIGNAL('plotWindowChanged(const QString&, const QString&)'), QtCore.QString(self.mooseObject.path + '/' + field), QtCore.QString(value))
            except KeyError:
                ret = False
        if ret:
            self.emit(QtCore.SIGNAL('dataChanged(const QModelIndex&, const QModelIndex&)'), index, index)
        return ret
                
        
    def data(self, index, role=Qt.DisplayRole):
        """Return the data  stored at given index.

        """
        if not index.isValid() or index.row() >= len(self.fields):
            return None
        ret = None
        field = self.fields[index.row()]        
        #if role == Qt.ToolTipRole:
            #print "$$",field,str(field)
            #return self.tr('<html>' + moose.doc(self.mooseObject.class_ + '.' + str(field)).replace(chr(27) + '[1m', '<b>').replace(chr(27) + '[0m', '</b>') + '</html>') # This is to remove special characters used for pretty printing in terminals
         #   return self.tr('<html>' + moose.doc(self.mooseObject.class_ + '.' + str(field)) + '</html>') # This is to remove special characters used for pretty printing in terminals
        if index.column() == 0 and role == Qt.DisplayRole:
            ret = QtCore.QVariant(QtCore.QString(field))
        elif index.column() == 1 and (role == Qt.DisplayRole or role == Qt.EditRole):
            try:
                field = ObjectFieldsModel.py_moose_fieldname_map[field]
            except KeyError:
                pass
            
            ret = self.mooseObject.getField(field)
            #print 'Field', field, 'value', ret
            ret = QtCore.QVariant(QtCore.QString(str(ret)))            
        elif index.column() == 2 and role == Qt.DisplayRole:
            try:
                ret = QtCore.QVariant(self.fieldPlotNameMap[field])
            except KeyError:
                pass
        return ret

    
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QtCore.QVariant(self._header[section])
        else:
            return None

    def flags(self, index):
        """
        Return the flags for the item pointed to by index.
        
        first column is field-name and is enabled and selectable
        
        second column is field value and its flags are precalculated
        in __init__ method and saved in self.fieldFlags

        third column is check button determining if the field is to be
        plotted.
        
        """
        flag = Qt.ItemIsEnabled
        if index.isValid():
            try:
                flag = self.fieldFlags[self.fields[index.row()]]
            except KeyError:
                flag = Qt.ItemIsEnabled
            if index.column() == 0:
                flag = flag & ~Qt.ItemIsEditable
            elif index.column() == 2:
                try:
                    flag = self.fieldPlotNameMap[self.fields[index.row()]]
                    if flag is not None:
                        flag = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
                except KeyError:
                    flag = Qt.ItemIsEnabled
        return flag

    def rowCount(self, parent):
        return len(self.fields)

    def columnCount(self, parent):
        return len(self._header)

    @property
    def checkedFields(self):
        checked_fields = []
        for field in self.fields:
            if self.fieldChecked[field]:
                checked_fields.append(field)
        return checked_fields

    def updatePlotField(self, index, plotWindowName):
        self.fieldPlotNameMap[self.fields[index.row()]] = str(plotWindowName)                
        self.emit(QtCore.SIGNAL('dataChanged(const QModelIndex&, const QModelIndex&)'), index, index)

        
class ObjectEditDelegate(QtGui.QItemDelegate):
    """Delegate to handle object editor"""
    def __init__(self, *args):
        QtGui.QItemDelegate.__init__(self, *args)

    def createEditor(self, parent, option, index):
        """Override createEditor from parent class to show custom
        combo box for the plot column."""
        widget = None
        if index.column() == 2:
            combobox = QtGui.QComboBox(parent)  
            combobox.addItems(index.model().plotNames)
            combobox.setEditable(False)
            self.index = index
            self.connect(combobox, QtCore.SIGNAL('currentIndexChanged( int )'), self.emitComboSelectionCommit)
            widget = combobox
        else:
            widget = QtGui.QItemDelegate.createEditor(self, parent, option, index)
        widget.setFocusPolicy(Qt.StrongFocus)
        return widget

    def emitComboSelectionCommit(self, index):
        if not isinstance(self.sender(), QtGui.QComboBox):
            raise TypeError('This should have never been reached. Only the plot selection ComboBox should be connected to this signal. But got: %s' % (self.sender()))
        self.emit(QtCore.SIGNAL('commitData(QWidget *)'), self.sender())


    def setEditorData(self, editor, index):
        text = index.model().data(index, Qt.DisplayRole).toString()
        if index.column == 2:
            ii = editor.findText(text)
            if ii == -1:
                ii = 0
            editor.setCurrentIndex(ii)
        else:
            QtGui.QItemDelegate.setEditorData(self, editor, index)

    def setModelData(self, editor, model, index):
        if index.column() == 2:
            model.setData(index, QtCore.QVariant(editor.currentText()))
        else:
            QtGui.QItemDelegate.setModelData(self, editor, model, index)

class ObjectEditView(QtGui.QTableView):
    """Extension of QTableView in order to automate update of the plot field when a field is dragged and dropped on a plot"""
    def __init__(self, *args):
        QtGui.QTableView.__init__(self, *args)
        self.setEditTriggers(self.DoubleClicked | self.SelectedClicked | self.EditKeyPressed)
        
    def dataChanged(self, tl, br):
        QtGui.QTableView.dataChanged(self, tl, br)
        self.viewport().update()


if __name__ == '__main__':
    app = QtGui.QApplication([])
    mainWin = QtGui.QMainWindow()
    view = QtGui.QTableView(mainWin)
    mainWin.setCentralWidget(view)
    model = ObjectFieldsModel(moose.Compartment('c'))
    view.setModel(model)
    mainWin.show()
    sys.exit(app.exec_())
            


# 
# objectedit.py ends here
	
