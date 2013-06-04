import os
from PyQt4 import QtGui, QtCore,Qt
import config     
from mplugin import MoosePluginBase, EditorBase, EditorWidgetBase, PlotBase, RunBase

class DialogWidget(QtGui.QDialog):
    def __init__(self,parent=None):
	QtGui.QWidget.__init__(self, parent)
	layout = QtGui.QGridLayout()
        
	self.modelPathLabel = QtGui.QLabel('Create modelpath under')
	self.modelPathEdit =  QtGui.QLineEdit('')
        layout.addWidget(self.modelPathLabel, 1, 0)
        layout.addWidget(self.modelPathEdit, 1, 2)

	self.modelPluginLabel = QtGui.QLabel('Plugin')
        layout.addWidget(self.modelPluginLabel, 2, 0)
	self.submenu = QtGui.QComboBox()
	with open(os.path.join(config.MOOSE_GUI_DIR,
                               'plugins', 
                               'list.txt')) as lfile:
            self.pluginNames = [line.strip() for line in lfile]
            self.pluginNames = [name for name in self.pluginNames if name]
            
        self.submenu.addItems(self.pluginNames)
        layout.addWidget(self.submenu,2,2)
        
        self.buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel)
        layout.addWidget(self.buttonBox,3,2)
	self.connect(self.buttonBox, QtCore.SIGNAL('accepted()'), self.accept)
        self.connect(self.buttonBox, QtCore.SIGNAL('rejected()'), self.reject)
	
	self.setLayout(layout)
    
        
if __name__ == '__main__':
    app =QtGui.QApplication([])
    widget = DialogWidget()
    widget.setWindowTitle('New Model')
    widget.setMinimumSize(400, 200)
    widget.show()
    app.exec_()
        

