# MOOSEPropertyModel.py --- 
# 
# Filename: MOOSEPropertyModel.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Thu Apr 16 14:22:29 2009 (+0530)
# Version: 
# Last-Updated: Wed Jun 24 21:33:19 2009 (+0530)
#           By: subhasis ray
#     Update #: 62
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

import moose
import types
from PyQt4 import QtGui
from PyQt4.Qt import Qt
from PyQt4 import QtCore

class PropertyModel(QtCore.QAbstractTableModel):
    def __init__(self, mooseObject, parent=None):
        QtCore.QAbstractTableModel.__init__(self)
        self._header = ("Field", "Value")
        self.mooseObject = mooseObject
        self.fields = [] # This should hold key-value pairs of (fieldName, isEditable)
        classObj = eval('moose.' + self.mooseObject.className)
        fieldList = self.mooseObject.getFieldList(moose.VALUE)
        c_dict = classObj.__dict__        
        for item in fieldList: 
            self.fields.append((item, True))
            
        self.beginInsertRows(QtCore.QModelIndex(), 0, len(self.fields) - 1)
        self.insertRows(0, len(self.fields))
        self.endInsertRows()
            

    def setData(self, index, value, role=Qt.EditRole):
        """Set the value of field  specified by index.

        NOTE: If the user tries to put invalid values, the field is
        reset to default value. Old edit is lost."""
        if index.column() == 0 or not self.fields[index.row()][1]:
            return False
        value = str(value.toString())
        field = self.fields[index.row()][0]
        self.mooseObject.setField(field, value)
        if field == 'name':
            self.emit(QtCore.SIGNAL('objectNameChanged(const QString&)'), QtCore.QString(field)) 
        self.emit(QtCore.SIGNAL("dataChanged(const QModelIndex&, const QModelIndex&)"), index, index)
        return True

    def data(self, index, role=Qt.DisplayRole):
        if not role == QtCore.Qt.DisplayRole or not index.isValid() or index.row() > len(self.fields):
            return QtCore.QVariant()
        field = self.fields[index.row()][0]
        if index.column() == 0:
            return QtCore.QVariant(QtCore.QString(field))
        elif index.column() == 1:
            value = self.mooseObject.getField(field)
            return QtCore.QVariant(QtCore.QString(value))

        return QtCore.QVariant()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return QtCore.QVariant(self._header[section])
        return QtCore.QVariant()

    def flags(self, index):
        flag = Qt.ItemIsEnabled
        if not index.isValid() or index.column() == 0:
            return flag
        flag = flag | Qt.ItemIsSelectable
        if self.fields[index.row()][1]:
            flag = flag | Qt.ItemIsEditable
        return flag

    def rowCount(self, parent):
        return len(self.fields)

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
    model = PropertyModel(moose.Compartment('c'))
    view.setModel(model)
    mainWin.show()
    sys.exit(app.exec_())



# 
# MOOSEPropertyModel.py ends here
