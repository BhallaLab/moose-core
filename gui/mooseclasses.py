# MOOSEClasses.py
# 
# Filename: MOOSEClasses.py
# Description: This file contains the MOOSE class names divided 
#       into categories for easy incorporation into the toolbox.
# Author: subhasis ray
# Maintainer: 
# Created: Sun Apr 12 14:05:01 2009 (+0530)
# Version: 
# Last-Updated: Thu Dec 23 17:01:56 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 231
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
import moose
from PyQt4 import QtCore, QtGui

moose_category = ['base', 'builtins', 'biophysics', 'kinetics', 'device', 'robots']
moose_class = {'Neutral':'base', 
               'Tick':'base',

               # biophysics directory
                 'BinSynchan':'biophysics',
#                  'BioScan':'biophysics', # For hsolve internal use
                 'CaConc':'biophysics',
                 'Cell':'biophysics',
                 'Compartment':'biophysics',
                 'DifShell':'biophysics',
                 'GHK':'biophysics',
                 'HHChannel2D':'biophysics',
                 'HHChannel':'biophysics',
                 'HHGate2D':'biophysics',
                 'HHGate':'biophysics',
                 'IntFire':'biophysics',
                 'IzhikevichNrn':'biophysics',
                 'Mg_block':'biophysics',
                 # 'MMPump':'biophysics',
                 'Nernst':'biophysics',
                 'RandomSpike':'biophysics',
                 'script_out':'biophysics',
                 'SpikeGen':'biophysics',
                 'StochSynchan':'biophysics',
                 'SymCompartment':'biophysics',
                 'SynChan':'biophysics',
		 'NMDAChan':'biophysics',
                 'TauPump':'biophysics',
               # builtins directory
                 'AscFile':'builtins',
                 'Calculator':'builtins',
                 'Interpol2D':'builtins',
                 'Interpol':'builtins',
                 'Table':'builtins',
                 'TimeTable':'builtins',
               # device directory
                 'DiffAmp':'device',
                 'PIDController':'device',
                 'PulseGen':'device',
                 'RC':'device',
                 
               # kinetics directory
                 # 'KineticManager':'kinetics',
                 'CylPanel':'kinetics',
                 'DiskPanel':'kinetics',
                 'Enzyme':'kinetics',
                 'Geometry':'kinetics',
                 'HemispherePanel':'kinetics',
                 'KinCompt':'kinetics',
                 'KinPlaceHolder':'kinetics',
                 'MathFunc':'kinetics',
                 'Molecule':'kinetics',
                 'Panel':'kinetics',
                 # 'Particle':'kinetics',
                 'Reaction':'kinetics',
                 'RectPanel':'kinetics',
                 # 'SmoldynHub':'kinetics',
                 'SpherePanel':'kinetics',
                 'Surface':'kinetics',
                 'TriPanel':'kinetics',
               # robots directory
               'SigNeur': 'robots',
               'Adaptor': 'robots'
               }

class MooseClassItem(QtGui.QListWidgetItem):
    """Items in MOOSE classbox"""
    def __init__(self, item_name, category, parent):
        QtGui.QListWidgetItem.__init__(self, parent)
        self.setText(item_name)

class MooseClassWidget(QtGui.QToolBox):
    def __init__(self, *args):
        QtGui.QToolBox.__init__(self, *args)
        layout = self.layout()
        self._listWidget = {}
        self._page = {}
        self._item = []
        for (class_name, category) in moose_class.items():
            try:
                widget = self._listWidget[category]
            except KeyError, e:
                widget = QtGui.QListWidget(self)
                self._listWidget[category] = widget
                self.connect(widget, 
                         QtCore.SIGNAL('itemDoubleClicked(QListWidgetItem*)'), 
                         self.signalItemDoubleClicked)
                widget.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding))            
                # widget.setLayout(QtGui.QVBoxLayout())
                layout.addWidget(widget)
            item = MooseClassItem(class_name, category, widget)
            item.setToolTip(self.tr('<html>%s</html>' % (moose.PyMooseBase.getContext().description(class_name))))
            self._item.append(item)
        for category in moose_category:
            widget = self._listWidget[category]            
            self.addItem(widget, category)


    def getClassListWidget(self, category='all'):
        if category == 'all':
            return self._listWidget.itervalues()
        else:
            return [self._listWidget[category]]

    def signalItemDoubleClicked(self, item):
        className = item.text()
        print 'Creating instance of class', className
        self.emit(QtCore.SIGNAL('classNameDoubleClicked(PyQt_PyObject)'), className)

if __name__ == '__main__':
    app = QtGui.QApplication([])
    widget = MooseClassWidget()
    widget.show()
    app.exec_()
# 
# mooseclasses.py ends here
