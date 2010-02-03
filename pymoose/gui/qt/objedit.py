# objedit.py --- 
# 
# Filename: objedit.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Thu Apr 16 14:22:29 2009 (+0530)
# Version: 
# Last-Updated: Tue Feb  2 11:23:03 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 166
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# This implements a property editing model for MOOSE objects.
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
sys.path.append('/home/subha/src/moose/pymoose')
import moose
import types

from PyQt4 import QtGui
from PyQt4.Qt import Qt
from PyQt4 import QtCore

class ObjectFieldsModel(QtCore.QAbstractTableModel):
    extra_fields = ['parent', 'childList', 'fieldList']
    sys_fields = ['node', 'cpu', 'dataMem', 'msgMem']
    excluded_fields = set(extra_fields + sys_fields)
    def __init__(self, mooseObject, parent=None):
        QtCore.QAbstractTableModel.__init__(self)
        self._header = ("Field", "Value", "Plot")
        self.mooseObject = mooseObject
        classObj = eval('moose.' + self.mooseObject.className)
        fieldList = self.mooseObject.getFieldList(moose.VALUE)
        self.field = []
        self.field_flags = {} # This should hold key-value pairs of fieldName: isEditable
        for fieldName in fieldList:
            print fieldName
            if fieldName not in ObjectFieldsModel.excluded_fields:
                self.field.append(fieldName)
                flag = Qt.ItemIsEnabled | Qt.ItemIsSelectable
                try:
                    prop = eval('moose.' + self.mooseObject.__class__.__name__ + '.' + fieldName)
                    if (type(prop) is property) and prop.fset:
                        flag = flag | Qt.ItemIsEditable
                except (SyntaxError, AttributeError):
                    pass

                self.field_flags[fieldName] = [flag, False]

        self.beginInsertRows(QtCore.QModelIndex(), 0, len(self.field) - 1)
        self.insertRows(0, len(self.field))
        self.endInsertRows()
            

    def setData(self, index, value, role=Qt.EditRole):
        """Set the value of field  specified by index.

        NOTE: If the user tries to put invalid values, the field is
        reset to default value. Old edit is lost."""
        if index.column() == 0:# or not self.field[index.row()][1]:
            return False
        value = str(value.toString())
        
        field = self.field[index.row()]
        if index.column() == 1:
            self.mooseObject.setField(field, value)
            if field == 'name':
                self.emit(QtCore.SIGNAL('objectNameChanged(const QString&)'), QtCore.QString(field)) 
        else: # Column 2 is a boolean telling if the field is to be plotted
            self.field_flags[field][index.column()+1] = not self.field_flags[field][index.column()+1]
            self.emit(QtCore.SIGNAL('plotOptionToggled(const QString&)'), QtCore.QString(field))
        
        self.emit(QtCore.SIGNAL("dataChanged(const QModelIndex&, const QModelIndex&)"), index, index)
        return True

    def data(self, index, role=Qt.DisplayRole):
        if not role == QtCore.Qt.DisplayRole or not index.isValid() or index.row() > len(self.field):
            return QtCore.QVariant()
        field = self.field[index.row()]
        if index.column() == 0:
            return QtCore.QVariant(QtCore.QString(field))
        elif index.column() == 1:
            value = self.mooseObject.getField(field)
            return QtCore.QVariant(QtCore.QString(value))
        else:
            field_flags = self.field_flags[field]
            if len(field_flags) > 1:
                return QtCore.QVariant(field_flags[1])

        return QtCore.QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self._header[section])
        return QtCore.QVariant()

    def flags(self, index):
        if not index.isValid():
            return 0
        elif index.column() == 0:
            return Qt.ItemIsSelectable | Qt.ItemIsEnabled
        else:
            flag = self.field_flags[self.field[index.row()]][0]
            return flag

    def rowCount(self, parent):
        return len(self.field)

    def columnCount(self, parent):
        return len(self._header)

    



if __name__ == "__main__":
    import sys
    #sys.path.append("..")
    import moose
    app =  QtGui.QApplication([])
    mainWin = QtGui.QMainWindow()
    view = QtGui.QTableView()
    mainWin.setCentralWidget(view)
    model = ObjectFieldsModel(moose.Compartment('c'))
    view.setModel(model)
    mainWin.show()
    sys.exit(app.exec_())



# 
# objedit.py ends here
