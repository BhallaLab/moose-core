# channeleditor.py --- 
# 
# Filename: channeleditor.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Wed Jul 18 19:06:39 2012 (+0530)
# Version: 
# Last-Updated: Wed Jul 18 20:00:49 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 90
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# A widget for editing Hodgkin-Huxley type ion channels.
# 
# 

# Change log:
# 
# 
# 

# Code:

import sys
from PyQt4 import QtGui, QtCore
import numpy

class GateEditor(QtGui.QWidget):
    """Utility to edit gate equations.
    
    It provides two line edits to enter the alpha and beta equations
    directly.
    """
    def __init__(self, *args):
        QtGui.QWidget.__init__(self, *args)
        self.useVmButton = QtGui.QRadioButton('Use Vm', self)
        self.useVmButton.setChecked(True)
        self.inputPanel = QtGui.QFrame(self)
        self.minVmLabel = QtGui.QLabel('Minimum Vm', self)
        self.maxVmLabel = QtGui.QLabel('Maximum Vm', self)
        self.divsVmLabel = QtGui.QLabel('Number of divisions', self)
        self.minVmEdit = QtGui.QLineEdit(self)
        self.maxVmEdit = QtGui.QLineEdit(self)
        self.divsVmEdit = QtGui.QLineEdit(self)
        self.equation = '(A + B * Vm) / (C + exp((Vm + D)/F))'
        self.alphaLabel = QtGui.QLabel(u'Equation for forward rate  \u03B1 ', self)
        self.betaLabel = QtGui.QLabel(u'Equation for backward rate \u03B2', self)
        self.alphaEdit = QtGui.QLineEdit(self)
        self.betaEdit =  QtGui.QLineEdit(self)
        self.formCombo = QtGui.QComboBox(self)
        self.formCombo.addItem(u'm\u221E - \u03C4m')
        self.formCombo.addItem(u'\u03B1 - \u03B2')
        layout = QtGui.QGridLayout(self.inputPanel)
        layout.addWidget(self.minVmLabel, 0, 0)
        layout.addWidget(self.minVmEdit, 0, 1)
        layout.addWidget(self.maxVmLabel, 0, 3)
        layout.addWidget(self.maxVmEdit, 0, 4)
        layout.addWidget(self.divsVmLabel, 0, 6)
        layout.addWidget(self.divsVmEdit, 0, 7)
        layout.addWidget(self.formCombo, 1, 0, 1, 4)
        layout.addWidget(self.alphaLabel, 2, 0, 1, 2)
        layout.addWidget(self.alphaEdit, 2, 2, 1, 7)
        layout.addWidget(self.betaLabel, 3, 0, 1, 2)
        layout.addWidget(self.betaEdit, 3, 2, 1, 7)
        self.inputPanel.setLayout(layout)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.useVmButton)
        layout.addWidget(self.inputPanel)
        self.setLayout(layout)
        self.connect(self.useVmButton, QtCore.SIGNAL('toggled(bool)'), self.toggleInputPanel)
        
    def toggleInputPanel(self, on):
        self.inputPanel.setVisible(on)
        self.adjustSize()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    QtGui.qApp = app
    editor = GateEditor()
    editor.show()
    sys.exit(app.exec_())

# 
# channeleditor.py ends here
