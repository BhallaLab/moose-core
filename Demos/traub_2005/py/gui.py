# gui.py --- 
# 
# Filename: gui.py
# Description: 
# Author: 
# Maintainer: 
# Created: Fri Jul 12 11:53:50 2013 (+0530)
# Version: 
# Last-Updated: Thu Jul 18 12:42:02 2013 (+0530)
#           By: subha
#     Update #: 660
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
# in stead of "from matplotlib.figure import Figure" do -
# from matplotlib.pyplot import figure as Figure
# see: https://groups.google.com/forum/#!msg/networkx-discuss/lTVyrmFoURQ/SZNnTY1bSf8J
# but does not help after the first instance when displaying cell
from matplotlib.figure import Figure
from matplotlib import patches

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

import networkx as nx
import numpy as np
import sys
from cells import *
from cell_test_util import setup_current_step_model
import matplotlib.gridspec as gridspec
import moose
from moose import utils as mutils

simdt = 1e-6
plotdt = 1e-4

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
            self.rootEdit.returnPressed.connect(self.getUpdatedChannelListWidget)
            self.plotButton = QtGui.QPushButton('Plot selected channels')
            self.plotButton.clicked.connect(self.plotActInact)
            layout.addWidget(self.getUpdatedChannelListWidget())
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
        
    def getUpdatedChannelListWidget(self):
        try:
            self.channelListWidget.clear()
            self.channels = {}
        except AttributeError:            
            self.channelListWidget = QtGui.QListWidget(self)
            self.channelListWidget.setSelectionMode( QtGui.QAbstractItemView.ExtendedSelection)
        root = str(self.rootEdit.text())
        for chan in self.getChannels(root).values():
            self.channelListWidget.addItem(chan.name)
        self.update()
        return self.channelListWidget

    def getChannelListWidget(self):
        try:
            return self.channelListWidget
        except AttributeError:
            return self.getUpdatedChannelListWidget()

    def __plotGate(self, gate, mlabel='', taulabel=''):
        gate = moose.element(gate)
        a = np.asarray(gate.tableA)
        b = np.asarray(gate.tableB)
        m = a/(a+b)
        tau = 1/(a+b)
        v = np.linspace(gate.min, gate.max, len(m))
        self.mhaxes.plot(v, m, label='%s %s' % (gate.path, mlabel))
        self.tauaxes.plot(v, tau, label='%s %s' % (gate.path, taulabel))
        print 'Plotted', gate.path, 'vmin=', gate.min, 'vmax=', gate.max, 'm[0]=', m[0], 'm[last]=', m[-1], 'tau[0]=', tau[0], 'tau[last]=', tau[-1]
        
    def plotActInact(self):
        """Plot the activation and inactivation variables of the selected channels"""
        try:
            self.figure.clear()
        except AttributeError:
            self.figure = Figure()
            self.canvas = FigureCanvas(self.figure)
            self.nav = NavigationToolbar(self.canvas, self)
            self.plotWidget = QtGui.QWidget()
            layout = QtGui.QVBoxLayout()
            self.plotWidget.setLayout(layout)
            layout.addWidget(self.canvas)
            layout.addWidget(self.nav)
        self.mhaxes = self.figure.add_subplot(2, 1, 1)
        self.mhaxes.set_title('Activation/Inactivation')
        self.tauaxes = self.figure.add_subplot(2, 1, 2)
        self.tauaxes.set_title('Tau')
        for item in self.getChannelListWidget().selectedItems():
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
        return self.plotWidget


from display_morphology import *

class NetworkXWidget(QtGui.QWidget):
    def __init__(self, *args, **kwargs):
        QtGui.QWidget.__init__(self, *args, **kwargs)
        layout = QtGui.QVBoxLayout()
        self.setLayout(layout)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.nav = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.nav)
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.axes.set_frame_on(False)
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)

    # TODO: bypass networkx draw as it uses pylab directly.
    def displayGraph(self, g, label=False):
        print g
        axon, sd = axon_dendrites(g)
        sizes = node_sizes(g) * 50
        weights = np.array([g.edge[e[0]][e[1]]['weight'] for e in g.edges()])
        pos = nx.graphviz_layout(g, prog='twopi')
        xmin, ymin, xmax, ymax = 1e9, 1e9, -1e9, -1e9
        for p in pos.values():
            if xmin > p[0]:
                xmin = p[0]
            if xmax < p[0]:
                xmax = p[0]
            if ymin > p[1]:
                ymin = p[1]
            if ymax < p[1]:
                ymax = p[1]        
        edge_widths = 10.0 * weights / max(weights)
        node_colors = ['k' if x in axon else 'gray' for x in g.nodes()]
        lw = [1 if n.endswith('comp_1') else 0 for n in g.nodes()]
        self.axes.clear()
        self.axes.set_xlim((xmin-10, xmax+10))
        self.axes.set_ylim((ymin-10, ymax+10))
        # print 'Cleared axes'
        for ii, e in enumerate(g.edges()):
            p0 = pos[e[0]]
            p1 = pos[e[1]]
            # print p0, p1
            a = patches.FancyArrow(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1], width=edge_widths[ii], head_width=0.0, axes=self.axes, ec='none', fc='black')
            # self.axes.plot(p0, p1)
            self.axes.add_patch(a)
            
            # a = self.axes.arrow(p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1], transform=self.figure.transFigure)
        # nx.draw_networkx_edges(g, pos, width=edge_widths, edge_color='gray', alpha=0.8, ax=self.axes)
        for ii, n in enumerate(g.nodes()):
            if n in axon:
                ec = 'black'
            elif n.endswith('comp_1'):
                ec = 'red'
            else:
                ec = 'none'
                
            c = patches.Circle(pos[n], radius=sizes[ii], axes=self.axes, ec=ec, fc='gray', alpha=0.8)
            self.axes.add_patch(c)
        # nx.draw_networkx_nodes(g, pos, with_labels=False,
        #                        nnode_size=sizes,
        #                        node_color=node_colors,
        #                        linewidths=lw, 
        #                        alpha=0.8 ,
        #                        ax=self.axes)
        # if label:
        #     labels = dict([(n, g.node[n]['label']) for n in g.nodes()])
        #     nx.draw_networkx_labels(g, pos, labels=labels, ax=self.axes)
        self.canvas.draw()




class CellView(QtGui.QWidget):
    def __init__(self, *args, **kwargs):
        QtGui.QWidget.__init__(self, *args, **kwargs)
        self.cells = {}
        layout = QtGui.QGridLayout()
        self.setLayout(layout)
        layout.addWidget(self.getControlPanel(), 0, 0, 2, 1)
        layout.addWidget(self.getCellMorphologyWidget(), 0, 1, 1, 1)
        layout.addWidget(self.getPlotWidget(), 1, 1, 1, 1)

    def getControlPanel(self):
        try:
            return self.controlPanel
        except AttributeError:
            self.controlPanel = QtGui.QWidget()
            layout = QtGui.QGridLayout()
            self.controlPanel.setLayout(layout)
            # self.rootEdit = QtGui.QLineEdit('/library')        
            # self.rootEdit.returnPressed.connect(self.getUpdatedCellListWidget)
            self.simtimeLabel = QtGui.QLabel('Simulate for (ms)')
            self.simtimeEdit = QtGui.QLineEdit('100')
            self.plotButton = QtGui.QPushButton('Simulate selected cell model')
            self.plotButton.clicked.connect(self.simulateSelected)
            layout.addWidget(self.getCellListWidget(), 0, 0, 1, 2)
            # layout.addWidget(self.rootEdit, 1, 0, 1, 1)
            layout.addWidget(self.getCurrentClampWidget(), 1, 0, 1, 2)
            layout.addWidget(self.simtimeLabel, 2, 0, 1, 1)
            layout.addWidget(self.simtimeEdit, 2, 1, 1, 1)
            layout.addWidget(self.plotButton, 3, 0, 1, 1)
            return self.controlPanel

    def getCells(self, root='/library'):
        if isinstance(root, str):
            root = moose.element(root)
        cells = []
        for cell in moose.wildcardFind('%s/#[ISA=Neuron]' % (root.path)):
            cells.append(cell[0].path.rpartition('/')[-1])
        print cells
        return cells
        
    def getUpdatedCellListWidget(self):
        try:
            self.cellListWidget.clear()
            self.cells = {}
        except AttributeError:            
            self.cellListWidget = QtGui.QListWidget(self)
            self.cellListWidget.itemSelectionChanged.connect(self.displaySelected)
            # self.cellListWidget.setSelectionMode( QtGui.QAbstractItemView.ExtendedSelection)
        # root = str(self.rootEdit.text())
        # print 'root = ', root
        for cell in self.getCells():
            self.cellListWidget.addItem(cell)     
        self.cellListWidget.setCurrentItem(self.cellListWidget.item(0))
        self.update()
        return self.cellListWidget

    def getCellListWidget(self):
        try:
            return self.cellListWidget
        except AttributeError:
            return self.getUpdatedCellListWidget()

    def getCurrentClampWidget(self):
        try:
            return self.currentClampWidget
        except AttributeError:
            self.currentClampWidget = QtGui.QWidget()
            self.delayLabel = QtGui.QLabel('Delay (ms)')
            self.delayText = QtGui.QLineEdit('20')
            self.widthLabel = QtGui.QLabel('Duration (ms)')
            self.widthText = QtGui.QLineEdit('50')
            self.ampLabel = QtGui.QLabel('Amplitude (pA)')
            self.ampText = QtGui.QLineEdit('100')
            title = QtGui.QLabel('Current pulse')
            layout = QtGui.QGridLayout()
            layout.addWidget(title, 0, 0, 1, 2)
            layout.addWidget(self.delayLabel, 1, 0, 1, 1)
            layout.addWidget(self.delayText, 1, 1, 1, 1)
            layout.addWidget(self.widthLabel, 2, 0, 1, 1)
            layout.addWidget(self.widthText, 2, 1, 1, 1)
            layout.addWidget(self.ampLabel, 3, 0, 1, 1)
            layout.addWidget(self.ampText, 3, 1, 1, 1)
            self.currentClampWidget.setLayout(layout)
        return self.currentClampWidget

    def getCellMorphologyWidget(self):
        try:
            return self.cellMorphologyWidget
        except AttributeError:
            self.cellMorphologyWidget = NetworkXWidget()
            return self.cellMorphologyWidget

    def displayCellMorphology(self, cellpath):
        cell = moose.element(cellpath)
        graph = cell_to_graph(cell)
        self.getCellMorphologyWidget().displayGraph(graph)

    def createCell(self, name):
        try:
            params = self.cells[name]
        except KeyError:
            model_container = moose.Neutral('/model_%s' % (name))
            data_container = moose.Neutral('/data_%s' % (name))
            params = setup_current_step_model(model_container,
                                              data_container,
                                              name,
                                              [[0, 0, 0],
                                               [1e9, 0, 0]])
            params['modelRoot'] = model_container.path
            params['dataRoot'] = data_container.path
            self.cells[name] = params
        return params

    def displaySelected(self):
        cellnames = [str(c.text()) for c in self.cellListWidget.selectedItems()]
        assert(len(cellnames) == 1)        
        name = cellnames[0]
        print 'Display', name
        params = self.createCell(name)
        # print params
        cell = params['cell']
        self.displayCellMorphology(cell.path)

    def simulateSelected(self):
        cellnames = [str(c.text()) for c in self.cellListWidget.selectedItems()]
        assert(len(cellnames) == 1)        
        name = cellnames[0]
        params = self.cells[name]
        try:
            scheduled = params['scheduled']
        except KeyError:
            hsolve = moose.HSolve('%s/solver' % (params['cell'].path))
            hsolve.dt = simdt
            hsolve.target = params['cell'].path
            mutils.setDefaultDt(elecdt=simdt, plotdt2=plotdt)
            mutils.assignDefaultTicks(modelRoot=params['modelRoot'],
                                      dataRoot=params['dataRoot'],
                                      solver='hsolve')
            params['scheduled'] = True
        delay = float(str(self.delayText.text()))
        width =  float(str(self.widthText.text()))
        level =  float(str(self.ampText.text()))
        params['stimulus'].delay[0] = delay * 1e-3
        params['stimulus'].width[0] = width * 1e-3
        params['stimulus'].level[0] = level * 1e-12
        moose.reinit()
        simtime = float(str(self.simtimeEdit.text()))*1e-3
        moose.start(simtime)         
        ts = np.linspace(0, simtime, len(params['somaVm'].vec))
        vm = params['somaVm'].vec
        stim = params['injectionCurrent'].vec
        self.vmAxes.clear()       
        self.vmAxes.set_title('membrane potential at soma')
        self.vmAxes.set_ylabel('mV')            
        self.vmAxes.plot(ts * 1e3, vm * 1e3, 'b-', label='Vm (mV)')
        self.vmAxes.get_xaxis().set_visible(False)
        self.stimAxes.clear()
        self.stimAxes.set_title('current injected at soma')
        self.stimAxes.set_ylabel('pA')
        self.stimAxes.set_xlabel('ms')
        self.stimAxes.plot(ts * 1e3, stim * 1e12, 'r-', label='Current (pA)')
        self.gs.tight_layout(self.plotFigure)
        # self.plotFigure.tight_layout()
        self.plotCanvas.draw()

    def getPlotWidget(self):
        try:
            return self.plotWidget
        except AttributeError:
            self.plotWidget = QtGui.QWidget()
            layout = QtGui.QVBoxLayout()
            self.plotFigure = Figure()
            self.plotCanvas = FigureCanvas(self.plotFigure)
            self.nav = NavigationToolbar(self.plotCanvas, self)
            self.gs = gridspec.GridSpec(3, 1)
            self.vmAxes = self.plotFigure.add_subplot(self.gs[:-1])
            self.vmAxes.set_frame_on(False)
            self.stimAxes = self.plotFigure.add_subplot(self.gs[-1], sharex=self.vmAxes)
            self.stimAxes.set_frame_on(False)
            self.gs.tight_layout(self.plotFigure)
            layout.addWidget(self.plotCanvas)
            layout.addWidget(self.nav)
            self.plotWidget.setLayout(layout)
        return self.plotWidget
                                              


if __name__ == '__main__':
    app = QtGui.QApplication([])
    # win = HHChanView()
    win = CellView()
    win.show()
    sys.exit(app.exec_())


# 
# gui.py ends here
