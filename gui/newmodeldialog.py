import os
from PyQt4 import QtGui, QtCore,Qt
import config     
from mplugin import MoosePluginBase, EditorBase, EditorWidgetBase, PlotBase, RunBase

class DialogWidget(QtGui.QDialog):
    def __init__(self,parent=None):
        self._currentRadioButton ="kkit"
        QtGui.QWidget.__init__(self, parent)
        
        
        layout = QtGui.QGridLayout()
        self.modelPathLabel = QtGui.QLabel('Model Name')
        self.modelPathEdit =  QtGui.QLineEdit('')
        layout.addWidget(self.modelPathLabel, 1, 0)
        layout.addWidget(self.modelPathEdit, 1, 1)
        #self.defaultRadio = QtGui.QRadioButton('default')
        #self.defaultRadio.setChecked(True);
        self.kkitRadio = QtGui.QRadioButton('kkit')
        self.kkitRadio.setChecked(True)
        #self.defaultRadio.toggled.connect(lambda : self.setcurrentRadioButton('default'))
        self.kkitRadio.toggled.connect(lambda : self.setcurrentRadioButton('kkit'))
        #layout.addWidget(self.defaultRadio,2,1)
        layout.addWidget(self.kkitRadio,2,1)
        self.buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        layout.addWidget(self.buttonBox,3,1)
        self.connect(self.buttonBox, QtCore.SIGNAL('accepted()'), self.accept)
        self.connect(self.buttonBox, QtCore.SIGNAL('rejected()'), self.reject)
        self.setLayout(layout)
    def setcurrentRadioButton(self,button):
        print " setcurrentRadioButton ",button
        self._currentRadioButton = button
    def getcurrentRadioButton(self):
        return self._currentRadioButton

if __name__ == '__main__':
    app =QtGui.QApplication([])
    widget = DialogWidget()
    widget.setWindowTitle('New Model')
    widget.setMinimumSize(400, 200)
    widget.show()
    app.exec_()
        

