from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys
import moose

#these will go to defaults.py
#these fields will be ignored
extra_fields = ['this','me','parent','path','class','children','linearSize','objectDimensions','lastDimension','localNumField','pathIndices','msgOut','msgIn','diffConst','speciesId','Coordinates','neighbors','DiffusionArea','DiffusionScaling','x','x0','x1','dx','nx','y','y0','y1','dy','ny','z','z0','z1','dz','nz','coords','isToroid','preserveNumEntries','numKm','numSubstrates','concK1']
#these fields denote the units of the field
field_units ={
    'Rm':'ohm'
}

def main():
    app = QApplication(sys.argv)
    w = MyWindow()
    w.show()
    sys.exit(app.exec_())

class MyWindow(QWidget):
    def __init__(self, *args):
        QWidget.__init__(self, *args)
        
        #c = moose.ZombiePool('/compartment')
        c = moose.Compartment('/compartment')

        tablemodel = MyTableModel(c,self) #my_array, self)
        tableview = QTableView()
        tableview.setModel(tablemodel)

        layout = QVBoxLayout(self)
        layout.addWidget(tableview)
        self.setLayout(layout)

class MyTableModel(QAbstractTableModel):
    def __init__(self, datain, parent=None, *args):
        QAbstractTableModel.__init__(self, parent, *args)
#        self.setAlternatingRowColors(True)
        self.fieldFlags = {}
        #self.arraydata = []
        self.fields = []
        self.mooseObject = datain
        
        for fieldName in self.mooseObject.getFieldNames('valueFinfo'):
            if(fieldName in extra_fields):
                continue
            else:
                value = self.mooseObject.getField(fieldName)
                
                #self.arraydata.append([fieldName+'(mV)',value])
                self.fields.append(fieldName)
            flag = Qt.ItemIsEnabled | Qt.ItemIsSelectable
            srchField = 'set_'+fieldName
            try:	
                for fn in (fn for fn in moose.getFieldDict(self.mooseObject.class_,'destFinfo').keys() if fn.startswith(srchField)):
                    flag = flag | Qt.ItemIsEditable
                    value = self.mooseObject.getField(fieldName)
                    print fieldName
            except Exception, e:
                print fieldName
                pass

#            print 'da'
            self.fieldFlags[fieldName] = flag
#            print 'fa'
#        print 'exiting init'

    def rowCount(self, parent):
        return len(self.fields)

    def columnCount(self, parent):
        return 2#len(self.fields)

    def setData(self, index, value, role=Qt.EditRole):
#        print 'setData Enter'
        oldValue = str(index.data().toString())
        
        if not index.isValid() and index.row () >= len(self.fields):
            return False
        ret = True
        value = (value.toString()) # convert Qt datastructure to
        if value =='':
            value = oldValue
        field = self.fields[index.row()]
        
        if index.column() == 0: # This is the fieldname
            ret = False
        elif index.column() == 1: # This is the value column
            if field == 'name':
                self.mooseObject.setField(field,str(value))
                #print 'something changed',str(value)
                self.emit(SIGNAL('objectNameChanged(PyQt_PyObject)'), self.mooseObject)
            else:    
                try: 
                    self.mooseObject.setField(field,float(value))
                except ValueError: #folks entering text instead of numerals here!
                    print "Numeric value should be entered";
                    pass
            
        if ret:
            self.emit(SIGNAL('dataChanged(const QModelIndex&, const QModelIndex&)'), index, index)
#        print 'setData Exit'
        return ret

    def flags(self, index):
#        print 'flags Enter'
        flag = Qt.ItemIsEnabled
        if index.isValid():
            try:
                flag = self.fieldFlags[self.fields[index.row()]]
            except KeyError:
                flag = Qt.ItemIsEnabled
            if index.column() == 0:
                flag = flag & ~Qt.ItemIsEditable
#        print 'flags Exit'
        return flag

    def data(self, index, role):
#        print 'data enter'
        ret = None
        field = self.fields[index.row()]
        if index.column() == 0 and role == Qt.DisplayRole:
            try:
                ret = QVariant(QString(field)+' ('+field_units[field]+')')
            except KeyError:
                ret = QVariant(QString(field))
        elif index.column() == 1 and (role == Qt.DisplayRole or role == Qt.EditRole):
            ret = self.mooseObject.getField(field)
            ret = QVariant(QString(str(ret)))
#        print 'data exit'
        return ret 

if __name__ == "__main__":
    main()
