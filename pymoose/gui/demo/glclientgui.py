# glclientgui.py --- 
# 
# Filename: glclientgui.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Sat Feb 13 16:01:54 2010 (+0530)
# Version: 
# Last-Updated: Sat Feb 13 16:07:22 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 10
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# This opens a GUI to start glclient and specify the parameters.
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

from PyQt4.Qt import Qt
from PyQt4 import QtGui, QtCore


class GLClientGUI(QtGui.QWidget):
    '''This is a GUI for setting the parameters for the glclient. The
    actual client process is created only after the parameters hae
    been set and the user pushes the "Start Client" button.'''
    def __init__(self, *args):
	QtGui.QWidget.__init__(self, *args)
	self.clientExecutablePath = config.GLCLIENT_EXECUTABLE
	


# 
# glclientgui.py ends here
