# test1.py --- 
# 
# Filename: test1.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Thu Jan 28 22:40:03 2010 (+0530)
# Version: 
# Last-Updated: Fri Jan 29 09:47:05 2010 (+0530)
#           By: subhasis ray
#     Update #: 50
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
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
from PyQt4 import QtCore, QtGui
from PyQt4.Qt import Qt

class MyPyShell(QtGui.QWidget):
    def __init__(self, parent=None):
	QtGui.QWidget.__init__(self, parent)
	self.textAreaWidget = QtGui.QTextEdit(self)
	self.textAreaWidget.setReadOnly(True)
	self.commandLineWidget = QtGui.QLineEdit(self)
	layout = QtGui.QVBoxLayout(self)
	layout.addWidget(self.textAreaWidget)
	layout.addWidget(self.commandLineWidget)
	self.setLayout(layout)

	self.connect(self.commandLineWidget, QtCore.SIGNAL('returnPressed()'), self.runCommand)

    def runCommand(self):
	cmd_text = str(self.commandLineWidget.text())
        print 'globals:'
        for (key, value) in globals().items():
            print key, ':', value
	print 'Running command:', cmd_text
	try:
	    exec cmd_text in globals()
	except Exception, e:
	    print 'Exeception', e
	    command = compile( cmd_text, '<string>', 'eval')
	    exec(command)
	finally:
	    self.commandLineWidget.clear()
            for (key, value) in globals().items():
                print key, ':', value


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    widget = MyPyShell()
    widget.show()
    app.exec_()


# 
# test1.py ends here
