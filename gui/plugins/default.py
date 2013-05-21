# default.py --- 
# 
# Filename: default.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Tue Nov 13 15:58:31 2012 (+0530)
# Version: 
# Last-Updated: Tue May 21 18:05:55 2013 (+0530)
#           By: subha
#     Update #: 1154
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# The default placeholder plugin for MOOSE
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

from collections import defaultdict
import numpy as np

from PyQt4 import QtGui, QtCore
from PyQt4.Qt import Qt
import moose
import mtree

from mplugin import MoosePluginBase, EditorBase, EditorWidgetBase, PlotBase, RunBase

class MoosePlugin(MoosePluginBase):
    """Default plugin for MOOSE GUI"""
    def __init__(self, root, mainwindow):
        MoosePluginBase.__init__(self, root, mainwindow)

    def getPreviousPlugin(self):
        return None

    def getNextPlugin(self):
        return None

    def getAdjacentPlugins(self):
        return []

    def getViews(self):
        return self._views

    def getCurrentView(self):
        return self.currentView

    def getEditorView(self):
        if not hasattr(self, 'editorView'):
            self.editorView = MooseEditorView(self)
            self.editorView.getCentralWidget().editObject.connect(self.mainWindow.objectEditSlot)
            self.currentView = self.editorView
        return self.editorView

    def getRunView(self):
        if not hasattr(self, 'runView') or self.runView is None:
            self.runView = RunView(self)
        return self.runView

    def getMenus(self):
        """Create a custom set of menus."""
        return self._menus


class MooseEditorView(EditorBase):
    """Default editor.

    """
    def __init__(self, plugin):
        EditorBase.__init__(self, plugin)
        self.__initMenus()
        self.__initToolBars()

    def __initMenus(self):
        editMenu = QtGui.QMenu('&Edit')
        for menu in self.getCentralWidget().getMenus():
            editMenu.addMenu(menu)
        self._menus.append(editMenu)

    def __initToolBars(self):
        for toolbar in self.getCentralWidget().getToolBars():
            self._toolBars.append(toolbar)

    def getToolPanes(self):
        return super(MooseEditorView, self).getToolPanes()

    def getLibraryPane(self):
        return super(MooseEditorView, self).getLibraryPane()

    def getOperationsWidget(self):
        return super(MooseEditorView, self).getOperationsPane()

    def getCentralWidget(self):
        """Retrieve or initialize the central widget.

        Note that we call the widget's setModelRoot() function
        explicitly with the plugin's modelRoot as the argument. This
        enforces an update of the widget display with the current
        modelRoot.

        This function should be overridden by any derived class as it
        has the editor widget class hard coded into it.

        """
        if self._centralWidget is None:
            self._centralWidget = DefaultEditorWidget()
            if hasattr(self._centralWidget, 'init'):
                self._centralWidget.init()
            self._centralWidget.setModelRoot(self.plugin.modelRoot)
        return self._centralWidget


class DefaultEditorWidget(EditorWidgetBase):
    """Editor widget for default plugin. 
    
    Plugin-writers should code there own editor widgets derived from
    EditorWidgetBase.

    Signals: editObject - emitted with currently selected element's
    path as argument. Should be connected to whatever slot is
    responsible for firing the object editor in top level.

    """
    editObject = QtCore.pyqtSignal('PyQt_PyObject')
    def __init__(self, *args):
        EditorWidgetBase.__init__(self, *args)
        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)

    def init(self):
        if hasattr(self, 'tree'):
            return
        self.tree = mtree.MooseTreeWidget()
        self.getTreeMenu()
        self._toolBars.append(QtGui.QToolBar('Insert'))
        for action in self.insertMenu.actions():
            self._toolBars[-1].addAction(action)
        self.layout().addWidget(self.tree)        

    def getTreeMenu(self):
        if hasattr(self, 'treeMenu'):
            return self.treeMenu
        self.treeMenu = QtGui.QMenu()
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.treeMenu.exec_)
        # Inserting a child element
        self.insertMenu = QtGui.QMenu('Insert')
        self._menus.append(self.insertMenu)
        self.treeMenu.addMenu(self.insertMenu)
        self.insertMapper = QtCore.QSignalMapper(self)
        for mclass in moose.element('/classes').children:
            action = QtGui.QAction(mclass.name[0], self.insertMenu)
            self.insertMenu.addAction(action)
            # NOTE: Signal mapping is sensitive to data types
            # in Python. It HAS TO BE OLD STYLE connect in PyQT. You
            # cannot get away with:
            #
            # action.triggered.connect(self.insertMapper.map)
            # or self.insertMapper.mapped.connect(self.tree.insertElementSlot)
            self.insertMapper.setMapping(action, QtCore.QString(mclass.name[0]))
            self.connect(action, QtCore.SIGNAL('triggered()'), self.insertMapper, QtCore.SLOT('map()'))
        self.connect(self.insertMapper, QtCore.SIGNAL('mapped(const QString&)'), self.tree.insertElementSlot)
        self.editAction = QtGui.QAction('Edit', self.treeMenu)
        self.editAction.triggered.connect(self.objectEditSlot)
        self.tree.elementInserted.connect(self.elementInsertedSlot)
        self.treeMenu.addAction(self.editAction)
        return self.treeMenu

    # def getMenus(self):
    #     return self._menus

    def updateModelView(self):
        current = self.tree.currentItem().mobj
        self.tree.recreateTree(root=self.modelRoot)
        self.tree.setCurrentItem(current)

    def updateItemSlot(self, mobj):
        """This should be overridden by derived classes to connect appropriate
        slot for updating the display item."""
        self.tree.updateItemSlot(mobj)

    def objectEditSlot(self):
        """Emits an `editObject(str)` signal with moose element path of currently selected tree item as
        argument"""
        mobj = self.tree.currentItem().mobj
        self.editObject.emit(mobj.path)

    def elementInsertedSlot(self, mobj):
        self.editObject.emit(mobj.path)        

    def sizeHint(self):
        return QtCore.QSize(400, 300)



from mplot import CanvasWidget

class RunView(RunBase):
    """A default runtime view implementation. This should be
    sufficient for most common usage.
    
    canvas: widget for plotting

    dataRoot: location of data tables

    """
    def __init__(self, *args, **kwargs):
        RunBase.__init__(self, *args, **kwargs)
        self.canvas = PlotWidget()
        self.modelRoot = self.plugin.modelRoot
        self.dataRoot = '%s/data' % (self.modelRoot)
        self.canvas.setModelRoot(self.modelRoot)
        self.canvas.setDataRoot(self.dataRoot)

    def getCentralWidget(self):
        """TODO: replace this with an option for multiple canvas
        tabs"""
        return self.canvas
        
    def setDataRoot(self, path):
        self.dataRoot = path

    def setModelRoot(self, path):
        self.modelRoot = path

    def getDataTablesPane(self):
        """This should create a tree widget with dataRoot as the root
        to allow visual selection of data tables for plotting."""
        raise NotImplementedError()

    def plotAllData(self):
        """This is wrapper over the same function in PlotWidget."""
        self.canvas.plotAllData()

    def getToolPanes(self):
        if not self._toolPanes:
            self._toolPanes = [self.getSchedulingDockWidget()]
        return self._toolPanes

    def getSchedulingDockWidget(self):
        """Create and/or return a widget for schduling"""
        if hasattr(self, 'schedulingDockWidget')  and self.schedulingDockWidget is not None:
            return self.schedulingDockWidget
        self.schedulingDockWidget = QtGui.QDockWidget('Scheduling')
        widget = SchedulingWidget()
        self.schedulingDockWidget.setWidget(widget)
        QtCore.QObject.connect(widget.runner, QtCore.SIGNAL('update'), self.canvas.updatePlots)
        QtCore.QObject.connect(widget.runner, QtCore.SIGNAL('finished'), self.canvas.rescalePlots)
        widget.resetAndRunButton.clicked.connect(self.canvas.plotAllData) 
        QtCore.QObject.connect(widget, QtCore.SIGNAL('simtimeExtended'), self.canvas.extendXAxes)
        return self.schedulingDockWidget


class MooseRunner(QtCore.QObject):
    """Helper class to control simulation execution

    See: http://doc.qt.digia.com/qq/qq27-responsive-guis.html :
    'Solving a Problem Step by Step' for design details.
    """
    def __init__(self, *args, **kwargs):
        QtCore.QObject.__init__(self, *args, **kwargs)
        self._updateInterval = 100e-3
        self._simtime = 0.0        
        self._clock = moose.Clock('/clock')
        self._pause = False

    def resetAndRun(self, tickDtMap, tickTargetMap, simtime, updateInterval):
        self._pause = False
        self._updateInterval = updateInterval
        self._simtime = simtime
        self.updateTicks(tickDtMap)
        self.assignTicks(tickTargetMap)
        QtCore.QTimer.singleShot(0, self.run)
        
    def run(self):
        """Run simulation for a small interval."""
        if self._clock.currentTime >= self._simtime:
            self.emit(QtCore.SIGNAL('finished'))
            return
        if self._pause:
            return
        toRun = self._simtime - self._clock.currentTime
        if toRun > self._updateInterval:
            toRun = self._updateInterval
        moose.start(toRun)
        self.emit(QtCore.SIGNAL('update'))
        QtCore.QTimer.singleShot(0, self.run)
    
    def continueRun(self, simtime, updateInterval):
        """Continue running without reset for `simtime`."""
        self._simtime = simtime
        self._updateInterval = updateInterval
        self._pause = False
        QtCore.QTimer.singleShot(0, self.run)
    
    def unpause(self):
        """Run for the rest of current simtime"""
        self._pause = False
        QtCore.QTimer.singleShot(0, self.run)

    def stop(self):
        """Pause simulation"""
        self._pause = True

    def updateTicks(self, tickDtMap):
        for tickNo, dt in tickDtMap.items():
            if tickNo >= 0 and dt > 0.0:
                moose.setClock(tickNo, dt)

    def assignTicks(self, tickTargetMap):
        for tickNo, target in tickTargetMap.items():
            moose.useClock(tickNo, target)


class SchedulingWidget(QtGui.QWidget):
    """Widget for scheduling.

    Important member fields:

    runner - object to run/pause/continue simulation. Whenevr an
    `updateInterval` time has been simulated this object sends an
    `update()` signal. This can be connected to other objects to
    update their data.

    """
    def __init__(self, *args, **kwargs):
        QtGui.QWidget.__init__(self, *args, **kwargs)
        layout = QtGui.QVBoxLayout()
        self.simtimeWidget = self.__getSimtimeWidget()
        self.tickListWidget = self.__getTickListWidget()
        self.runControlWidget = self.__getRunControlWidget()
        layout.addWidget(self.runControlWidget)
        layout.addWidget(self.simtimeWidget)
        layout.addWidget(self.tickListWidget)
        self.updateInterval = 100e-3 # This will be made configurable with a textbox
        layout.addWidget(self.__getUpdateIntervalWidget())
        self.setLayout(layout)
        self.runner = MooseRunner()
        self.connect(self, QtCore.SIGNAL('resetAndRun'), self.runner.resetAndRun)
        self.resetAndRunButton.clicked.connect(self.resetAndRun)
        self.continueButton.clicked.connect(self.continueRun)
        self.connect(self, QtCore.SIGNAL('continueRun'), self.runner.continueRun)
        self.runTillEndButton.clicked.connect(self.runner.unpause)
        self.stopButton.clicked.connect(self.runner.stop)        
        self.connect(self.runner, QtCore.SIGNAL('update'), self.updateCurrentTime)

    def __getUpdateIntervalWidget(self):
        label = QtGui.QLabel('Plot update interval')
        self.updateIntervalText = QtGui.QLineEdit(str(self.updateInterval))
        label = QtGui.QLabel('Update plots after every')
        ulabel = QtGui.QLabel('seconds of simulation')
        layout = QtGui.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.updateIntervalText)
        layout.addWidget(ulabel)
        widget = QtGui.QWidget()
        widget.setLayout(layout)
        return widget

    def __getRunControlWidget(self):
        widget = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        self.resetAndRunButton = QtGui.QPushButton('Reset and run')
        self.stopButton = QtGui.QPushButton('Stop')
        self.continueButton = QtGui.QPushButton('Continue')
        self.runTillEndButton = QtGui.QPushButton('Run to end')
        layout.addWidget(self.resetAndRunButton)
        layout.addWidget(self.stopButton)
        layout.addWidget(self.runTillEndButton)
        layout.addWidget(self.continueButton)
        widget.setLayout(layout)
        return widget

    def updateUpdateInterval(self):
        """Read the updateInterval from text box.

        If updateInterval is less than the smallest dt, then make it
        equal.
        """
        try:
            self.updateInterval = float(str(self.updateIntervalText.text()))
        except ValueError:
            QtGui.QMessageBox.warning(self, 'Invalid value', 'Specified plot update interval is meaningless.')
        dt = min(self.getTickDtMap().values())
        if dt > self.updateInterval:
            self.updateInterval = dt
            self.updateIntervalText.setText(str(dt))
        
    def resetAndRun(self):
        """This is just for adding the arguments for the function
        MooseRunner.resetAndRun"""
        self.updateUpdateInterval()
        self.emit(QtCore.SIGNAL('simtimeExtended'), self.getSimTime())
        self.emit(QtCore.SIGNAL('resetAndRun'), 
                  self.getTickDtMap(), 
                  self.getTickTargets(), 
                  self.getSimTime(), 
                  self.updateInterval)

    def continueRun(self):
        """Helper function to emit signal with arguments"""
        self.updateUpdateInterval()
        self.emit(QtCore.SIGNAL('simtimeExtended'), 
                  self.getSimTime() + moose.Clock('/clock').currentTime)
        self.emit(QtCore.SIGNAL('continueRun'),
                  self.getSimTime(),
                  self.updateInterval)

    def __getSimtimeWidget(self):
        layout = QtGui.QGridLayout()
        simtimeWidget = QtGui.QWidget()
        self.simtimeEdit = QtGui.QLineEdit('1')
        self.currentTimeLabel = QtGui.QLabel('0')
        layout.addWidget(QtGui.QLabel('Run for'), 0, 0)
        layout.addWidget(self.simtimeEdit, 0, 1)
        layout.addWidget(QtGui.QLabel('seconds'), 0, 2)        
        layout.addWidget(QtGui.QLabel('Current time:'), 1, 0)
        layout.addWidget(self.currentTimeLabel, 1, 1)
        layout.addWidget(QtGui.QLabel('second'), 1, 2)        
        simtimeWidget.setLayout(layout)
        return simtimeWidget

    def __getTickListWidget(self):
        layout = QtGui.QGridLayout()
        # Set up the column titles
        layout.addWidget(QtGui.QLabel('Tick'), 0, 0)
        layout.addWidget(QtGui.QLabel('dt'), 0, 1)
        layout.addWidget(QtGui.QLabel('Targets (wildcard)'), 0, 2, 1, 2)
        layout.setRowStretch(0, 1)
        # Create one row for each tick. Somehow ticks.shape is
        # (16,) while only 10 valid ticks exist. The following is a hack
        ticks = moose.ematrix('/clock/tick')
        for ii in range(ticks[0].localNumField):
            tt = ticks[ii]
            layout.addWidget(QtGui.QLabel(tt.path), ii+1, 0)
            layout.addWidget(QtGui.QLineEdit(str(tt.dt)), ii+1, 1)
            layout.addWidget(QtGui.QLineEdit(''), ii+1, 2, 1, 2)
            layout.setRowStretch(ii+1, 1)            
        # We add spacer items to the last row so that expansion
        # happens at bottom. All other rows have stretch = 1, and
        # the last one has 0 (default) so that is the one that
        # grows
        rowcnt = layout.rowCount()
        for ii in range(3):
            layout.addItem(QtGui.QSpacerItem(1, 1), rowcnt, ii)
        layout.setRowStretch(rowcnt, 10)
        # layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 2)
        widget = QtGui.QWidget()
        widget.setLayout(layout)
        return widget

    def updateCurrentTime(self):
        self.currentTimeLabel.setText('%f' % (moose.Clock('/clock').currentTime))

    def updateTextFromTick(self, tickNo):
        tick = moose.ematrix('/clock/tick')[tickNo]
        widget = self.tickListWidget.layout().itemAtPosition(tickNo + 1, 1).widget()
        if widget is not None and isinstance(widget, QtGui.QLineEdit):
            widget.setText(str(tick.dt))

    def updateFromMoose(self):
        """Update the tick dt from the tick objects"""
        ticks = moose.ematrix('/clock/tick')
        # Items at position 0 are the column headers, hence ii+1
        for ii in range(ticks[0].localNumField):
            self.updateTextFromTick(ii)
        self.updateCurrentTime()

    def getSimTime(self):
        try:
            time = float(str(self.simtimeEdit.text()))
            return time
        except ValueError, e:
            QtGui.QMessageBox.warning(self, 'Invalid value', 'Specified runtime was meaningless.')
        return 0

    def getTickTargets(self):
        """Return a dictionary containing tick nos mapped to the
        target specified in tick list widget. If target is empty, the
        tick is not included."""
        ret = {}
        for ii in range(1, self.tickListWidget.layout().rowCount()):
            widget = self.tickListWidget.layout().itemAtPosition(ii, 2).widget()
            if widget is not None and isinstance(widget, QtGui.QLineEdit):
                target = str(widget.text()).strip()
                if len(target) > 0:
                    ret[ii-1] = target
        return ret

    def getTickDtMap(self):
        ret = {}
        # Items at position 0 are the column headers, hence ii+1
        for ii in range(1, self.tickListWidget.layout().rowCount()):
            widget = self.tickListWidget.layout().itemAtPosition(ii, 1).widget()
            if widget is not None and isinstance(widget, QtGui.QLineEdit):
                try:
                    ret[ii-1] = float(str(widget.text()))
                except ValueError:
                    QtGui.QMessageBox.warning(self, 'Invalid value', '`dt` for tick %d was meaningless.' % (ii-1))
        return ret
                             

from collections import namedtuple

# Keeps track of data sources for a plot. 'x' can be a table path or
# '/clock' to indicate time points from moose simulations (which will
# be created from currentTime field of the `/clock` element and the
# number of dat points in 'y'. 'y' should be a table. 'z' can be empty
# string or another table or something else. Will not be used most of
# the time (unless 3D or heatmap plotting).

PlotDataSource = namedtuple('PlotDataSource', ['x', 'y', 'z'], verbose=False)

class PlotWidget(CanvasWidget):
    """A wrapper over CanvasWidget to handle additional MOOSE-specific
    stuff.

    modelRoot - path to the entire model our plugin is handling

    dataRoot - path to the container of data tables.

    TODO: do we really need this separation or should we go for
    standardizing location of data with respect to model root.

    pathToLine - map from moose path to Line2D objects in plot. Can
    one moose table be plotted multiple times? Maybe yes (e.g., when
    you want multiple other tables to be compared with the same data).

    lineToPath - map from Line2D objects to moose paths

    """
    def __init__(self, *args, **kwargs):
        CanvasWidget.__init__(self, *args, **kwargs)
        self.modelRoot = '/'
        self.pathToLine = defaultdict(set)
        self.lineToPath = {}

    @property
    def plotAll(self):
        return len(self.pathToLine) == 0

    def setModelRoot(self, path):
        self.modelRoot = path

    def setDataRoot(self, path):
        self.dataRoot = path

    def plotAllData(self):
        path = self.dataRoot        
        time = moose.Clock('/clock').currentTime
        for tabId in moose.wildcardFind('%s/##[TYPE=Table]' % (path)):
            tab = moose.Table(tabId)
            if len(tab.neighbours['requestData']) > 0:
                # This is the default case: we do not plot the same
                # table twice. But in special cases we want to have
                # multiple variations of the same table on different
                # axes.
                #
                lines = self.pathToLine[tab.path]
                if len(lines) == 0:
                    newLines = self.addTimeSeries(tab, label=tab.name)
                    self.pathToLine[tab.path].update(newLines)
                    for line in newLines:
                        self.lineToPath[line] = PlotDataSource(x='/clock', y=tab.path, z='')
                else:
                    for line in lines:
                        ts = np.linspace(0, time, len(tab.vec))
                        line.set_data(ts, tab.vec)                             
        self.callAxesFn('legend')
        self.figure.canvas.draw()
                
    def addTimeSeries(self, table, *args, **kwargs):        
        ts = np.linspace(0, moose.Clock('/clock').currentTime, len(table.vec))
        return self.plot(ts, table.vec, *args, **kwargs)
        
    def addRasterPlot(self, eventtable, yoffset=0, *args, **kwargs):
        """Add raster plot of events in eventtable.

        yoffset - offset along Y-axis.
        """
        y = np.ones(len(eventtable.vec)) * yoffset
        return self.plot(eventtable.vec, y, '|')

    def updatePlots(self):
        for path, lines in self.pathToLine.items():            
            tab = moose.Table(path)
            data = tab.vec
            ts = np.linspace(0, moose.Clock('/clock').currentTime, len(data))
            for line in lines:
                line.set_data(ts, data)
        self.figure.canvas.draw()

    def extendXAxes(self, xlim):
        for axes in self.axes.values():
            axes.set_xlim(left=0, right=xlim)
        self.figure.canvas.draw()

    def rescalePlots(self):
        """This is to rescale plots at the end of simulation.
        
        TODO: ideally we should set xlim from simtime.
        """
        for axes in self.axes.values():
            axes.relim()
            axes.autoscale_view(True,True,True)
        self.figure.canvas.draw()
# 
# default.py ends here
