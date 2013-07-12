# gui.py --- 
# 
# Filename: gui.py
# Description: 
# Author: 
# Maintainer: 
# Created: Fri Jul 12 11:53:50 2013 (+0530)
# Version: 
# Last-Updated: Fri Jul 12 15:38:48 2013 (+0530)
#           By: subha
#     Update #: 174
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

"""
Display channel properties graphically
"""

from PyQt4 import QtCore, QtGui
from matplotlib import mlab
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

import moose

class HHChanView(QtGui.QWidget):
    def __init__(self, *args, **kwargs):
        QtGui.QWidget.__init__(self, *args, **kwargs)
        self.channels = {}
        self.setLayout(QtGui.QHBoxLayout())
        self.layout().addWidget(self.getControlPanel())
        self.layout().addWidget(self.plotActInact())        

    def getControlPanel(self):
        try:
            return self.controlPanel
        except AttributeError:
            self.controlPanel = QtGui.QWidget()
            layout = QtGui.QVBoxLayout()
            self.controlPanel.setLayout(layout)
            self.rootEdit = QtGui.QLineEdit('/library')        
            self.rootEdit.returnPressed.connect(self.getUpdatedChannelListView)
            self.plotButton = QtGui.QPushButton('Plot selected channels')
            self.plotButton.clicked.connect(self.plotActInact)
            layout.addWidget(self.getUpdatedChannelListView())
            layout.addWidget(self.rootEdit)
            layout.addWidget(self.plotButton)
            return self.controlPanel

    def getChannels(self, root='/library'):
        if isinstance(root, str):
            root = moose.element(root)
        for chan in moose.wildcardFind('%s/#[ISA=HHChannel]' % (root.path)):
            channel = moose.element(chan)
            self.channels[channel.name] = channel
        return self.channels
        
    def getUpdatedChannelListView(self):
        try:
            self.channelListView.clear()
            self.channels = {}
        except AttributeError:            
            self.channelListView = QtGui.QListWidget(self)
            self.channelListView.setSelectionMode( QtGui.QAbstractItemView.ExtendedSelection)
        root = str(self.rootEdit.text())
        for chan in self.getChannels(root).values():
            self.channelListView.addItem(chan.name)
        self.update()
        return self.channelListView

    def getChannelListView(self):
        try:
            return self.channelListView
        except AttributeError:
            return self.getUpdatedChannelListView()

    def __plotGate(self, gate, mlabel='', taulabel=''):
        gate = moose.element(gate)
        a = np.asarray(gate.tableA)
        b = np.asarray(gate.tableB)
        m = a/(a+b)
        tau = 1/(a+b)
        v = np.linspace(gate.min, gate.max, len(m))
        self.mhaxes.plot(v, m, label='%s %s' % (gate.path, mlabel))
        self.tauaxes.plot(v, tau, label='%s %s' % (gate.path, taulabel))
        
    def plotActInact(self):
        """Plot the activation and inactivation variables of the selected channels"""
        try:
            self.figure.clear()
        except AttributeError:
            self.figure = Figure()
            self.canvas = FigureCanvas(self.figure)
        self.mhaxes = self.figure.add_subplot(2, 1, 1)
        self.mhaxes.set_title('Activation/Inactivation')
        self.tauaxes = self.figure.add_subplot(2, 1, 2)
        self.tauaxes.set_title('Tau')
        for item in self.getChannelListView().selectedItems():
            chan = self.channels[str(item.text())]
            if chan.Xpower > 0:
                self.__plotGate(chan.gateX.path)
            if chan.Ypower > 0:
                self.__plotGate(chan.gateY.path)
            # if chan.Zpower > 0:
            #     self.__plotGate(chan.gateZ.path, mlabel='z', taulabel='tau-z')
        self.mhaxes.legend()
        self.tauaxes.legend()
        self.canvas.draw()
        return self.canvas

import sys
from cells import *

if __name__ == '__main__':
    app = QtGui.QApplication([])
    win = HHChanView()
    win.show()
    sys.exit(app.exec_())


# 
# gui.py ends here
