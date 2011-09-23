import sys
import pickle
import numpy
from PyQt4 import QtGui,QtCore
from PyQt4.Qt import Qt
import h5py
#from header import newGLWindow
from updatepaintGL import newGLWindow
from oglfunc.sobjects import *

def updateColor(w,q,t):
#    p = numpy.digitize(q[t],w.stepVals)
    
    for i in range(len(w.vizObjects)):
        w.vizObjects[i].r,w.vizObjects[i].g,w.vizObjects[i].b = w.colorMap[q[t][i]-1]
        w.vizObjects[i].radius = indRadius[r[t][i]-1]
    w.updateGL()

def getValue(value):
    updateColor(w,q,value)
    
def playMovie():
    t = newWin.slider.value()
    t += 1        
    if t < len(q):
        newWin.slider.setValue(t)
    else:
        stopTimer()
        
# def playMovie1():
#     t = newWin.slider.tickPosition()
#     while t<(len(q)-1):
#         time.sleep(0.2)
#         newWin.slider.setValue(t)
#         #w.rotate([1,0,0],50)
#         w.rotate([0,1,1],1)
#         w.translate([0,0.05,0])
#         updateColor(w,q,t)
#         pic = w.grabFrameBuffer()
#         pic.save('movie/sim_'+str(t)+'.png','PNG')
#         t += 1
#         #w.rotate([1,0,0],-50)

def startTimer():
    newWin.ctimer.start(50)

def stopTimer():
    newWin.ctimer.stop()

app = QtGui.QApplication(sys.argv) 
newWin = newGLWindow()
newWin.show()
newWin.setWindowState(Qt.WindowMaximized)

f = open(sys.argv[1],'r')
p = pickle.load(f)
f.close()

q = []
dummy=[]
for i in range(3060):
    dummy = []
    for j in range(3060):
        dummy.append(p[i+(j*10001)])
    q.append(dummy)

indRadius = numpy.arange(0.05,0.20,0.005)

#timer
newWin.ctimer = QtCore.QTimer()
newWin.connect(newWin.ctimer, QtCore.SIGNAL("timeout()"),playMovie)

#toolbar
newWin.toolbar = QtGui.QToolBar()
newWin.toolbar.setMinimumHeight(30)
newWin.addToolBar(Qt.BottomToolBarArea,newWin.toolbar)

#play
newWin.playButton = QtGui.QToolButton(newWin.toolbar)
newWin.playButton.setIcon(QtGui.QIcon('play.png'))
newWin.playButton.setGeometry(0,0,30,30)
newWin.connect(newWin.playButton, QtCore.SIGNAL('clicked()'),startTimer)

#pause
newWin.pauseButton = QtGui.QToolButton(newWin.toolbar)
newWin.pauseButton.setIcon(QtGui.QIcon('pause.png'))
newWin.pauseButton.setGeometry(30,0,30,30)
newWin.connect(newWin.pauseButton, QtCore.SIGNAL('clicked()'),stopTimer)

#slider
newWin.slider = QtGui.QSlider(QtCore.Qt.Horizontal,newWin.centralwidget)
newWin.slider.setRange(0,len(q)-2)
newWin.slider.setTickPosition(0)
newWin.connect(newWin.slider, QtCore.SIGNAL('valueChanged(int)'), getValue)

#default setting
#newWin.verticalLayout.addWidget(newWin.toolbar)
newWin.verticalLayout.addWidget(newWin.slider)
newWin.setWindowState(Qt.WindowMaximized)

w =  newWin.mgl
i = 0
for yAxis in range(60):
    for xAxis in range(60):
        if i < 3060:
            comp = somaDisk(parent=w,cellName=str(i),l_coords=[0,0,0,0,0,0,0,str(i)])
            comp.radius = 0.20
            comp._centralPos=[xAxis*0.5,yAxis*0.5,0.0]
            #w.sceneObjects.append(comp)
            #w.sceneObjectNames.append(str(i))
            w.vizObjects.append(comp)
            #w.vizObjectNames.append(str(i))
            i += 1

w.setColorMap()
w.setColorMap_2()
w.updateGL()

print 'done drawing'

r=[0]*len(q)
for i in range(len(q)):
    q[i] = numpy.digitize(q[i],w.stepVals)
    r[i] = numpy.digitize(q[i],w.stepVals_2)

# f = h5py.File(sys.argv[1],'r')
# group = f['/Vm']
# vm = None

# for cellName in group.keys():
#     tmp = numpy.zeros(shape=group[cellName].shape, dtype='float64')
#     tmp[:] = group[cellName][:]
#     if vm is None:
#         vm = tmp
#     else:
#         vm = numpy.hstack((vm, tmp))
# print 'reading done'

# f = open('vm.pkl','w')
# pickle.dump(vm,f)
# f.close()

# for i in range(len(vm)):
#         q.append(numpy.digitize(vm[i],w.stepVals))

# print len(q)
# print 'digitizing'         

sys.exit(app.exec_())
