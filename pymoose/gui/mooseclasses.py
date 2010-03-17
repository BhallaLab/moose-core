# MOOSEClasses.py
# 
# Filename: MOOSEClasses.py
# Description: This file contains the MOOSE class names divided 
#       into categories for easy incorporation into the toolbox.
# Author: subhasis ray
# Maintainer: 
# Created: Sun Apr 12 14:05:01 2009 (+0530)
# Version: 
# Last-Updated: Wed Mar 17 10:28:07 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 43
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
#sys.path.append('..')
import moose
from PyQt4 import QtCore, QtGui

class MooseClasses:
    """This class roughly reflects the directory structure in MOOSE source code.

    The primary purpose of this class is to divide the MOOSE classes
    into categories.  Each category will form a page in the GUI
    toolbox and the list of classes under that category will be
    displayed on that page. The list "category" is used to order the
    pages in toolbox."""
    categories = ["base", "builtins", "biophysics", "kinetics", "device"]
    class_dict = {
        "base":[
                "Neutral", 
                "Tick"],
        "biophysics": [
                "BinSynchan",
                "BioScan",
                "CaConc",
                "Cell",
                "Compartment",
                "DifShell",
                "GHK",
                "HHChannel2D",
                "HHChannel",
                "HHGate2D",
                "HHGate",
                "IntFire",
                "IzhikevichNrn",
                "Mg_block",
                "MMPump",
                "Nernst",
                "RandomSpike",
                "script_out",
                "SpikeGen",
                "StochSynchan",
                "SymCompartment",
                "SynChan",
                "NMDAChan",
                "TauPump",
                "TestBiophysics"],
        "builtins":[
                "AscFile",
                "Calculator",
                "Interpol2D",
                "Interpol",
                "Table",
                "TimeTable"],
        "device": [
                "DiffAmp",
                "PIDController",
                "PulseGen",
                "RC"],
        "kinetics": [
                "KineticManager",
                "CylPanel",
                "DiskPanel",
                "Enzyme",
                "Geometry",
                "HemispherePanel",
                "KinCompt",
                "KinPlaceHolder",
                "MathFunc",
                "Molecule",
                "Panel",
                "Particle",
                "Reaction",
                "RectPanel",
                "SmoldynHub",
                "SpherePanel",
                "Surface",
                "TriPanel"]}
class MooseClassItem(QtGui.QListWidgetItem):
    """Items in MOOSE classbox"""
    def __init__(self, item_name, category, parent):
        QtGui.QListWidgetItem.__init__(self, parent)
        self.setText(item_name)

class MooseClassWidget(QtGui.QToolBox):
    def __init__(self, *args):
        QtGui.QToolBox.__init__(self, *args)
        self.listWidgets = []
        self.pages = []
        for category in MooseClasses.categories:
            page = QtGui.QWidget(self)
            page.setObjectName(category + "Page")
            page.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding))
            self.pages.append(page)
            layout = QtGui.QVBoxLayout(page)
            page.setLayout(layout)
            class_list = QtGui.QListWidget(page)
            class_list.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding))
            layout.addWidget(class_list)
            self.listWidgets.append(class_list)
            for item_name in MooseClasses.class_dict[category]:
                class_item = MooseClassItem(item_name, category, class_list)
                class_item.setToolTip(self.tr(moose.PyMooseBase.getContext().description(item_name)))
            self.addItem(page, category)

if __name__ == '__main__':
    app = QtGui.QApplication([])
    widget = MooseClassWidget()
    widget.show()
    app.exec_()
# 
# mooseclasses.py ends here
