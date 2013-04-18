# objectedit.py ---
#
# Filename: objectedit.py
# Description:
# Author: Subhasis Ray
# Maintainer:
# Created: Wed Jun 30 11:18:34 2010 (+0530) 
# Version:
# Last-Updated: Thu Apr 18 19:18:46 2013 (+0530)
#           By: subha
#     Update #: 589
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
# Wed Jun 30 11:18:34 2010 (+0530) - Originally created by Subhasis
# Ray, the model and the view 
#
# Modified/adapted to dh_branch by Chaitanya/Harsharani
#
# Thu Apr 18 18:37:31 IST 2013 - Reintroduced into multiscale GUI by
# Subhasis
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

from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys
sys.path.append('../python')
import moose
import defaults

#these fields will be ignored
extra_fields = ['this',
                'me',
                'parent',
                'path',
                # 'class',
                'children',
                'linearSize',
                'objectDimensions',
                'lastDimension',
                'localNumField',
                'pathIndices',
                'msgOut',
                'msgIn',
                'diffConst',
                'speciesId',
                'Coordinates',
                'neighbors',
                'DiffusionArea',
                'DiffusionScaling',
                'x',
                'x0',
                'x1',
                'dx',
                'nx',
                'y',
                'y0',
                'y1',
                'dy',
                'ny',
                'z',
                'z0',
                'z1',
                'dz',
                'nz',
                'coords',
                'isToroid',
                'preserveNumEntries',
                'numKm',
                'numSubstrates',
                'concK1']

class ObjectEditModel(QAbstractTableModel):
    def __init__(self, datain, headerdata=['Field','Value'], parent=None, *args):
        QAbstractTableModel.__init__(self, parent, *args)
        self.fieldFlags = {}
        self.fields = []
        self.mooseObject = datain
        self.headerdata = headerdata
        self.undoStack = []        
        for fieldName in self.mooseObject.getFieldNames('valueFinfo'):            
            if fieldName in extra_fields :
                continue
            value = self.mooseObject.getField(fieldName)
            self.fields.append(fieldName)
        flag = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        # !! This is outrageous !! - Subha
        # searchField = 'set_'+fieldName
        # try:	
        #     for fn in (fn for fn in moose.getFieldDict(self.mooseObject.class_,'destFinfo').keys() if fn.startswith(srchField)):
        #         flag = flag | Qt.ItemIsEditable
        #         value = self.mooseObject.getField(fieldName)
        # except Exception, e:
        #     pass
        # !! end outrageous !!
        self.fieldFlags[fieldName] = flag

    def rowCount(self, parent):
        return len(self.fields)

    def columnCount(self, parent):
        return len(self.headerdata)

    def setData(self, index, value, role=Qt.EditRole):
        oldValue = str(index.data().toString())        
        if not index.isValid() and index.row () >= len(self.fields):
            return False
        ret = True
        value = str(value.toString()) # convert Qt datastructure to Python string
        if value =='':
            value = oldValue
        field = self.fields[index.row()]        
        if index.column() == 0: # This is the fieldname
            ret = False
        elif index.column() == 1: # This is the value column
            if field == 'name':
                self.mooseObject.setField(field,str(value))
                self.emit(SIGNAL('objectNameChanged(PyQt_PyObject)'), self.mooseObject)
            else:    
                try: 
                    self.mooseObject.setField(field,float(value))
                except ValueError: #folks entering text instead of numerals here!
                    print "Numeric value should be entered";
                    pass            
        if ret:
            self.emit(SIGNAL('dataChanged(const QModelIndex&, const QModelIndex&)'), index, index)
        return ret

    def flags(self, index):
        flag =  Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if not index.isValid():
            return None
        # Replacing the `outrageous` up stuff with something sensible
        setter = 'set_%s' % (self.fields[index.row()])
        if index.column() == 1 and setter in self.mooseObject.getFieldNames('destFinfo'):
            flag |= Qt.ItemIsEditable
        # !! Replaced till here
        return flag

    def data(self, index, role):
        ret = None
        field = self.fields[index.row()]
        if index.column() == 0 and role == Qt.DisplayRole:
            try:
                ret = QVariant(QString(field)+' ('+defaults.FIELD_UNITS[field]+')')
            except KeyError:
                ret = QVariant(QString(field))
        elif index.column() == 1 and (role == Qt.DisplayRole or role == Qt.EditRole):
            try:
                ret = self.mooseObject.getField(str(field))
                ret = QVariant(QString(str(ret)))
            except ValueError:
                ret = None
        return ret 

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.headerdata[col])
        return QVariant()

class ObjectEditView(QTableView):
    def __init__(self, *args):
        QTableView.__init__(self, *args)
        #self.setEditTriggers(self.DoubleClicked | self.SelectedClicked | self.EditKeyPressed)
        vh = self.verticalHeader()
        vh.setVisible(False)

        hh = self.horizontalHeader()
        hh.setStretchLastSection(True)

        self.setAlternatingRowColors(True)
        self.resizeColumnsToContents()

    def dataChanged(self, tl, br):
        QTableView.dataChanged(self, tl, br)
        self.viewport().update()

def main():
    app = QApplication(sys.argv)
    c = moose.Compartment('test_compartment')
    model = ObjectEditModel(c)
    view = ObjectEditView()
    view.setModel(model)
    view.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
#
