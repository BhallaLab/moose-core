import sys
from PyQt4 import QtGui
from PyQt4.Qt import Qt
from updatepaintGL import newGLWindow


app = QtGui.QApplication(sys.argv) 
newWin = newGLWindow()
newWin.show()
newWin.setWindowState(Qt.WindowMaximized)
widget =  newWin.mgl



sys.exit(app.exec_())
