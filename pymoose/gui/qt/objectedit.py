# objectedit.py --- 
# 
# Filename: objectedit.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed Jun 30 11:18:34 2010 (+0530)
# Version: 
# Last-Updated: Fri Jul  2 11:13:46 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 271
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
    
    extra_fields -- list of fields that are of no use in the fields editor.

    sys_fields -- list of fields that carry system information.

    """
    extra_fields = ['parent', 'childList', 'fieldList']
    sys_fields = ['node', 'cpu', 'dataMem', 'msgMem']

    def __init__(self, mooseObject, parent=None):
        """Set up the model. 

        The table model has one moose field in each row.  A field that
        has a set method is editable. Fields listed in extra_fields
        are not shown.

        """
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._header = ('Field', 'Value', 'Plot')
        self.mooseObject = mooseObject
        self.fields = []
        self.fieldFlags = {}
        self.fieldCheckFlags = {}
        self.fieldChecked = {}
        try:
            classObject = eval('moose.' + self.mooseObject.className)
        except AttributeError:
            return

        for fieldName in self.mooseObject.getFieldList(moose.VALUE):
            if fieldName in ObjectFieldsModel.extra_fields:
                continue
            flag = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            checkFlag = Qt.ItemIsEnabled
            try:
                prop = eval('moose.' + self.mooseObject.__class__.__name__ + '.' + fieldName)
                if (type(prop) is property) and prop.fset:
                    flag = flag | Qt.ItemIsEditable
                value = mooseObject.getField(fieldName)
                try:
                    dummy = float(value)
                    checkFlag = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
                except ValueError:
                    pass
                    
            except SyntaxError, se:
                config.LOGGER.error('%s' % (str(se)))
            except AttributeError, ae:
                config.LOGGER.error('%s' % (str(ae)))
            self.fieldFlags[fieldName] = flag
            self.fieldCheckFlags[fieldName] = checkFlag
            self.fieldChecked[fieldName] = False
            self.fields.append(fieldName)
            
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
            self.mooseObject.setField(field, value)
            if field == 'name':
                self.emit(QtCore.SIGNAL('objectNameChanged(const QString&)'), QtCore.QString(field))
        elif index.column() == 2 and role == Qt.CheckStateRole: # Checkbox for plotting
            if self.fieldCheckFlags[self.fields[index.row()]] & Qt.ItemIsUserCheckable: # This field is checkable
                self.fieldChecked[field] = not self.fieldChecked[field]
                self.emit(QtCore.SIGNAL('plotOptionToggled(const QString&)'), QtCore.QString(field))
            else:
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
        if index.column() == 0 and role == Qt.DisplayRole:
            ret = QtCore.QVariant(QtCore.QString(field))
        elif index.column() == 1 and role == Qt.DisplayRole:
            ret = QtCore.QVariant(QtCore.QString(self.mooseObject.getField(field)))
        elif index.column() == 2 and role == Qt.CheckStateRole and (self.fieldCheckFlags[field] & Qt.ItemIsUserCheckable):
            ret = QtCore.QVariant(self.fieldChecked[field])
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
            if index.column() == 0:
                flag = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            elif index.column() == 1:
                try:
                    flag = self.fieldFlags[self.fields[index.row()]]
                except KeyError, e:
                    pass
            elif index.column() == 2 and self.fieldCheckFlags[self.fields[index.row()]]:
                try:
                    flag = self.fieldCheckFlags[self.fields[index.row()]]
                except KeyError:
                    pass
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
