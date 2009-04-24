# MOOSEWorkspace.py --- 
# 
# Filename: MOOSEWorkspace.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Sun Apr 12 17:12:47 2009 (+0530)
# Version: 
# Last-Updated: Sun Apr 12 22:08:45 2009 (+0530)
#           By: subhasis ray
#     Update #: 49
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

from PyQt4 import QtGui

from MOOSEElement import *

class MOOSEWorkspace(QtGui.QGroupBox):
    def __init__(self, *args):
        QtGui.QGroupBox.__init__(self, *args)
        self.current_widget = self
        self.class_instance_dict = {}
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        self.setAcceptDrops(True)
        

    def addElementSlot(self, item):
        class_name = item.text()
        count = 0
        try:
            count = self.class_instance_dict[class_name]
        except KeyError:
            pass
        count = count + 1
        self.class_instance_dict[class_name] = count
        new_element = MOOSEElement(class_name + str(count), self.current_widget)
        self.layout().addWidget(new_element)
        print "Created", new_element.title()
        

# 
# MOOSEWorkspace.py ends here
