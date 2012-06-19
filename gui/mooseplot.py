import sys 
from PyQt4 import QtGui, QtCore
from PyQt4.Qt import Qt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy import array,linspace
import config
import moose
import os

class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        self.xmin = 100.1

        self.axes = fig.add_subplot(111)
        self.axes.set_navigate(True)
        self.axes.hold(True)
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Value')

        self.compute_initial_figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

class MoosePlot(MyMplCanvas):
    plot_index = 0
    
    def __init__(self,*args,**kwargs):
        MyMplCanvas.__init__(self,*args,**kwargs)

        self.plotNo = MoosePlot.plot_index
        MoosePlot.plot_index += 1
        self.setAcceptDrops(True)
        self.curveIndex = 0
        self.curveTableMap = {}
        self.tableCurveMap = {}
        self.mpl_connect('pick_event',self.onpick)
        
    def onpick(self,event):
        ind=event.ind[0]
        print self._labels[ind]
        return True

    def addTable(self,table,curve_name=None):
        try: #checks if table is already added
            curve = self.tableCurveMap[table]
        except KeyError:
            print 'Adding Table ',table
            if curve_name is None:
                curve_name = table.getField('path')
            curve, =  self.axes.plot([0],[0],label=curve_name)
            self.curveTableMap[curve] = table
            self.tableCurveMap[table] = curve
        if len(table.vec) > 0:
            yy = array(table.vec)
            xx = linspace(0.0,self.xmin,len(yy))
            curve.set_data(xx[1:len(xx)],yy[1:len(yy)])
            
        self.axes.relim()
        self.axes.autoscale_view(True,True,True)
        self.axes.legend()
        self.axes.figure.canvas.draw()

    def updatePlot(self, currentTime):
        config.LOGGER.debug('update: %g' % (currentTime))
        if currentTime > self.xmin:
            self.xmin = currentTime
        for curve, table in self.curveTableMap.items():
            tabLen = len(table)
            if tabLen == 0:
                continue
            ydata = array(table.table)           
            xdata = linspace(0, currentTime, tabLen)
            curve.set_data([xdata[2:tabLen:1]],[ydata[2:tabLen:1]])

        self.axes.relim()
        self.axes.autoscale_view(True,True,True)
        self.axes.figure.canvas.draw()

    def reset(self):
        table_list = []
        try:
            while self.tableCurveMap:
                (table, curve) = self.tableCurveMap.popitem()
                self.curveTableMap.pop(curve)
                table_list.append(table)
        except KeyError:
            pass
        for table in table_list:
            self.addTable(table)

    def removeTable(self, table):
        try:
            curve = self.tableCurveMap.pop(table)
            curve.remove()
            self.curveTableMap.pop(curve)
        except KeyError:
            pass

    def savePlotData(self, directory=''):
        for table in self.tableCurveMap.keys():
            filename = os.path.join(directory, table.name + '.plot')
            print 'Saving', filename
            table.dumpFile(filename)

class MoosePlotWindow(QtGui.QMainWindow):

    def __init__(self, *args):
        QtGui.QMainWindow.__init__(self, *args)

        self.resize(567, 497)
        self.centralwidget = QtGui.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.plot = MoosePlot(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot.sizePolicy().hasHeightForWidth())
        self.plot.setSizePolicy(sizePolicy)
        self.plot.setObjectName("plot")
        self.verticalLayout.addWidget(self.plot)
        self.setCentralWidget(self.centralwidget)
        
    def closeEvent(self, event):
        self.emit(QtCore.SIGNAL('windowClosed()'))
        self.hide()
        

import sys
if __name__ == '__main__':
    app = QtGui.QApplication([])
    testComp = moose.Compartment('c')
    testTable = moose.Table('t')
    moose.connect(testTable,'requestData', testComp, 'get_Vm')

    testPulse = moose.PulseGen('p')
    testPulse.firstDelay = 50e-3
    testPulse.firstWidth = 40e-3
    testPulse.firstLevel = 1e-9
    moose.connect(testPulse,'outputOut', testComp, 'injectMsg')

    simdt = 1e-4/4

    moose.setClock(0, simdt)
    moose.setClock(1, simdt)
    moose.setClock(3, simdt)
    
    moose.useClock(0, 'c' , 'init')
    moose.useClock(0, 'p' , 'process')
    moose.useClock(1, 'c', 'process')
    moose.useClock(3, 't', 'process')
    moose.reinit()
    moose.start(0.30)
 
    plotWin = MoosePlotWindow()
    plotWin.plot.addTable(testTable)
    plotWin.show()
        
    sys.exit(app.exec_())
        
