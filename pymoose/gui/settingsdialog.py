from ui_settings_dialog import *
from PyQt4 import QtCore, QtGui

class SettingsDialog(QtGui.QDialog, Ui_settingsDialog):
    def __init__(self, *args):
	QtGui.QDialog.__init__(self, *args)
	self.setupUi(self)
	
    def simPathList(self):
	txtdoc = self.simPathTextEdit.document()
	txt = txtdoc.toPlainText()
	pathList = str(txt).split('\n')
	return pathList
	
if __name__ == '__main__':
	app = QtGui.QApplication([])
	dialog = SettingsDialog()
	dialog.show()
	app.exec_()
