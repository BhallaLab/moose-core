from PyQt4 import QtGui, QtCore

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

class MyMplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        self.axes = fig.add_subplot(111)
        self.axes.set_navigate(True)
        self.axes.hold(True)
        self.axes.clear()
        self.compute_initial_figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        # FigureCanvas.setSizePolicy(self,
        #                            QtGui.QSizePolicy.Expanding,
        #                            QtGui.QSizePolicy.Expanding)
        # FigureCanvas.updateGeometry(self)

    def update_graph(self,y,name):
        self.axes.plot(y,label=name)
        self.axes.set_xlabel('time')

        fontP = FontProperties()
        fontP.set_size('small')
        box = self.axes.get_position()
        self.axes.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])

        self.axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5,prop=fontP)

        self.draw()

    def grid_on(self):
        self.axes.get_xaxis().grid(True)
        self.axes.get_yaxis().grid(True)

    def compute_initial_figure(self):
        pass


class MplWidget(MyMplCanvas):

    def __init__(self,*args,**kwargs):
        MyMplCanvas.__init__(self,*args,**kwargs)
	self.mpl_connect('pick_event',self.onpick)

    def onpick(self,event):
        ind=event.ind[0]
        print self._labels[ind]
        return True
	
	

	
