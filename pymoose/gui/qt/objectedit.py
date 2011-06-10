# objectedit.py --- 
# 
# Filename: objectedit.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed Jun 30 11:18:34 2010 (+0530)
# Version: 
# Last-Updated: Fri Jun 10 11:24:14 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 511
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

class ObjectFieldsModel(QtCore.QAbstractTableModel):
    """Model the fields list for MOOSE objects.
    
    extra_fields -- list of fields that are of no use in the fields
                    editor.

    sys_fields -- list of fields that carry system information. This
                  is for future - so that we can restrict the
                  visibility of these fields to advanced mode.

    """
    extra_fields = ['parent', 'childList', 'fieldList', 'index', 'xtree_textfg_req', 'xtree_fg_req','nInitComplex','concInitComplex', 'step_mode', 'tableVector']
    sys_fields = ['node', 'cpu', 'dataMem', 'msgMem', 'class']
    moose_py_fieldname_map = {'step_mode': 'stepMode',
                              'stepmode': 'stepMode',
                              'lambda': 'lambda_',
                              'calc_mode': 'calcMode',
                              'abs_refract':'absRefractT',
                              'stepsize': 'stepSize'}
    py_moose_fieldname_map = {'stepMode': 'step_mode',
                              'calcMode': 'calc_mode',
                              'lambda_': 'lambda',
                              'absRefractT': 'abs_refract',
                              'stepSize': 'stepsize'
                              }
    
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
        self._header = ('Field', 'Value', 'Plot')
        self.fields = []
        self.plotNames = ['None']
        self.fieldFlags = {}
        self.fieldPlotNameMap = {}
        try:
            className = 'moose.' + mooseObject.className
            config.LOGGER.debug('Creating editor model for %s of class %s' % (mooseObject.path, className))
            classObject = eval(className)
            self.mooseObject = classObject(mooseObject.id)

        except AttributeError:
            config.LOGGER.error('Could not wrap object %s into class %s' % (mooseObject.path, className))
            return

        for fieldName in self.mooseObject.getFieldList(moose.FTYPE_VALUE):
            config.LOGGER.debug('class: %s, python class: %s, path: %s, field: %s' % (self.mooseObject.className, self.mooseObject.__class__.__name__, self.mooseObject.path, fieldName))
            if (fieldName in ObjectFieldsModel.extra_fields) or (fieldName in ObjectFieldsModel.sys_fields):
                continue
            if fieldName in ObjectFieldsModel.moose_py_fieldname_map.keys():
                pyFieldName = ObjectFieldsModel.moose_py_fieldname_map[fieldName]
            else:
                pyFieldName = fieldName
            flag = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            try:
                prop = eval('moose.' + self.mooseObject.__class__.__name__ + '.' + pyFieldName)
                if (type(prop) is property) and prop.fset:
                    flag = flag | Qt.ItemIsEditable
                value = mooseObject.getField(fieldName)
                try:
                    dummy = float(value)
                    flag = flag | Qt.ItemIsDragEnabled
                    self.fieldPlotNameMap[fieldName] = self.plotNames[0]
                except ValueError:
                    pass
            except Exception, e:
                config.LOGGER.error("%s" % (e))

            self.fieldFlags[pyFieldName] = flag
            self.fields.append(pyFieldName)            
        self.insertRows(0, len(self.fields))

    def setData(self, index, value, role=Qt.EditRole):
        """Set field value or set plot flag.

        If a user tries to put an invalid value then the field is
        reset to default value. Old edit is lost.
        """
        if not index.isValid() and index.row () >= len(self.fields):
            return False
        ret = True
        value = str(value.toString()) # convert Qt datastructure to
                                      # Python datastructure

        field = self.fields[index.row()]
        if index.column() == 0: # This is the fieldname
            ret = False
        elif index.column() == 1: # This is the value column
            try:
                field = ObjectFieldsModel.py_moose_fieldname_map[field]
            except KeyError:
                pass
            self.mooseObject.setField(field, value)
            if field == 'name':
                self.emit(QtCore.SIGNAL('objectNameChanged(PyQt_PyObject)'), self.mooseObject)
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
        if role == Qt.ToolTipRole:
            return self.tr('<html>' + moose.context.doc(self.mooseObject.className + '.' + str(field)).replace(chr(27) + '[1m', '<b>').replace(chr(27) + '[0m', '</b>') + '</html>') # This is to remove special characters used for pretty printing in terminals
        if index.column() == 0 and role == Qt.DisplayRole:
            ret = QtCore.QVariant(QtCore.QString(field))
        elif index.column() == 1 and role == Qt.DisplayRole:
            config.LOGGER.debug('Field: %s' % (field))
            try:
                field = ObjectFieldsModel.py_moose_fieldname_map[field]
            except KeyError:
                pass
            ret = QtCore.QVariant(QtCore.QString(self.mooseObject.getField(field)))
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
        # print 'Creating editor'
        
        if index.column() == 2:
            combobox = QtGui.QComboBox(parent)        
            combobox.addItems(index.model().plotNames)
            combobox.setEditable(False)
            self.index = index
            self.connect(combobox, QtCore.SIGNAL('currentIndexChanged( int )'), self.emitComboSelectionCommit)
        
            # print 'create Combobox'
            return combobox
        return QtGui.QItemDelegate.createEditor(self, parent, option, index)

    def emitComboSelectionCommit(self, index):
        if not isinstance(self.sender(), QtGui.QComboBox):
            raise TypeError('This should have never been reached. Only the plot selection ComboBox should be connected to this signal. But got: %s' % (self.sender()))
#        self.emit(QtCore.SIGNAL('commitData(QWidget *)'), self.sender())
#        self.emit(QtCore.SIGNAL('plotValueChanged(int,QModelIndex * )'),index,self.index)
#        self.emit(QtCore.SIGNAL('plotWindowChanged(const QString&, const QString&)'), QtCore.QString(self.mooseObject.path + '/' + field), QtCore.QString(value))

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
