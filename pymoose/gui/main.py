# main.py --- 
# 
# Filename: main.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Tue Jun 16 11:12:18 2009 (+0530)
# Version: 
# Last-Updated: Tue Jun 16 16:38:10 2009 (+0530)
#           By: subhasis ray
#     Update #: 33
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Simple GUI for loading models in MOOSE
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
from PyQt4.Qt import Qt
from PyQt4 import QtCore, QtGui

from mainwin import MainWindow


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    QtCore.QObject.connect(app, QtCore.SIGNAL('lastWindowClosed()'), app, QtCore.SLOT('quit()'))
    mainwin = MainWindow()
    mainwin.show()
    app.exec_()



# 
# main.py ends here
