import sys 
from PyQt4 import QtGui, QtCore
from PyQt4.Qt import Qt

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from numpy import array,linspace
import config
import moose
import os

class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_size_inches(width,height,forward=True)

        self.xmin = 100.1

        self.axes = self.fig.add_subplot(111)
        self.axes.set_navigate(True)
        self.axes.hold(True)
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Value')
        #selfax = plt.subplot(111)
        self.compute_initial_figure()
        self.canvas = FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        self.listOfFewColors = ['b','r','g','c','m','y','purple']
        self.listOfLineStyles = ['-','--','-.',':']
        
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)



    def sizeHint(self):
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    def minimumSizeHint(self):
        return QtCore.QSize(10, 10)

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
        self.prevCurves = []
        self.oldCurves = []
        self.overlayPlots = False
        self.alreadyOverlayed = False #check for already overlayed
        self.alreadyPlaced = False

    def nicePlaceLegend(self):
        box = self.axes.get_position()
        if self.alreadyPlaced == False:
            self.axes.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            self.alreadyPlaced = True
        # Put a legend below current axis
        self.axes.legend(loc='upper center',prop={'size':10}, bbox_to_anchor=(0.5, -0.083),fancybox=True, shadow=True, ncol=4)

    def changeToOverlay(self):
        #print 'overlay on'
        self.overlayPlots = True
        self.oldCurves.extend(self.curveTableMap.keys()) # save the existing curves

    def clearOldLines(self):
        #print 'overlay off'
        self.oldCurves = []

        for line in self.prevCurves:
            line.pop(0).remove()
        self.prevCurves = []

        self.overlayPlots = False
        self.alreadyOverlayed = False

    def onpick(self,event):
        ind=event.ind[0]
        print self._labels[ind]
        return True

    def addTable(self,table,curve_name=None,lineColor=None):
        if lineColor == None:
            lineColor = self.listOfFewColors[self.plotNo % len(self.listOfFewColors)]
        try: #checks if table is already added
            curve = self.tableCurveMap[table]
        except KeyError:
            if curve_name is None:
                curve_name = table.getField('path')
            curve, =  self.axes.plot([0],[0],label=curve_name,color=lineColor)
            self.curveTableMap[curve] = table
            self.tableCurveMap[table] = curve
        if len(table.vec) > 0:
            yy = array(table.vec)
            xx = linspace(0.0,self.xmin,len(yy))
            curve.set_data(xx[0:len(xx)],yy[0:len(yy)])
            
        self.plotNo += 1
        self.axes.relim()
        self.axes.autoscale_view(True,True,True)
        self.axes.legend()
        self.axes.figure.canvas.draw()

    def updatePlot(self, currentTime):
        config.LOGGER.debug('update: %g' % (currentTime))
        #print 'updateplots in mooseplot'

        if self.overlayPlots and not self.alreadyOverlayed :
            self.alreadyOverlayed = True
            for curve in self.oldCurves:
                l = self.axes.plot(curve.get_data()[0][0],curve.get_data()[1][0],label=str(curve.get_label())+'_old',ls='--',color=curve.get_color())
                self.prevCurves.append(l)
            self.nicePlaceLegend()
                    
        if currentTime > self.xmin:
            self.xmin = currentTime
        for curve, table in self.curveTableMap.items():
            tabLen = len(table.vec)
            if tabLen == 0:
                continue
            ydata = array(table.vec)           
            xdata = linspace(0, currentTime, tabLen)
            curve.set_data([xdata[0:tabLen:1]],[ydata[0:tabLen:1]])
            curve.set_linestyle('-')


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

class MoosePlotWindow(QtGui.QMdiSubWindow):
    """This is to customize MDI sub window for our purpose.

    In particular, we don't want anything to be deleted when the window is closed. 
    
    """
    def __init__(self, *args):
        QtGui.QMdiSubWindow.__init__(self, *args)
        
    def closeEvent(self, event):
        self.emit(QtCore.SIGNAL('subWindowClosed()'))
        self.hide()


class newPlotSubWindow(QtGui.QMdiSubWindow):

    def __init__(self, *args):
        QtGui.QMdiSubWindow.__init__(self, *args)
        self.plot = MoosePlot(self,width=6,height=6)
        self.plot.setObjectName("plot")

        l = self.layout()
        l.addWidget(self.plot)

        qToolBar = QtGui.QToolBar()
        self.toolbar = NavigationToolbar(self.plot, qToolBar)
        qToolBar.addWidget(self.toolbar)
        qToolBar.setMovable(False)
        qToolBar.setFloatable(False)
        l.addWidget(qToolBar)
        
    def closeEvent(self, event):
        self.emit(QtCore.SIGNAL('subWindowClosed()'))
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
        
