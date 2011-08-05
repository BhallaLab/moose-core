import sys
from PyQt4 import QtGui,QtCore
from PyQt4.Qt import Qt
import time#, sched
import pickle
from updatepaintGL import newGLWindow

from oglfunc.objects import *
from oglfunc.group import *

#s = sched.scheduler(time.time, time.sleep)

def updateColor(w,q,t):
    for i in range(len(w.vizObjects)):
        w.vizObjects[i].r,w.vizObjects[i].g,w.vizObjects[i].b = w.colorMap[q[t][i]-1]
    w.updateGL()

def getValue(value):
    #print int
    updateColor(w,q,value)

def playMovie():
    t = newWin.slider.tickPosition()
#    newWin.playButton.setEnabled(False)
    while t<(len(q)-1):
        
        time.sleep(0.2)
        newWin.slider.setValue(t)
        updateColor(w,q,t)
        t += 1
        
#    newWin.playButton.setEnabled(True)

def playMovie1():
    t = newWin.slider.tickPosition()
    while t<(len(q)-1):
        time.sleep(0.2)
        newWin.slider.setValue(t)
        #w.rotate([1,0,0],50)
        w.rotate([0,1,1],1)
        w.translate([0,0.05,0])
        updateColor(w,q,t)
        pic = w.grabFrameBuffer()
        pic.save('movie/sim_'+str(t)+'.png','PNG')
        t += 1
        #w.rotate([1,0,0],-50)

app = QtGui.QApplication(sys.argv) 
newWin = newGLWindow()
newWin.show()
newWin.setWindowState(Qt.WindowMaximized)

f = open('tada.pkl','r')
q = pickle.load(f)
f.close()



newWin.playButton =  QtGui.QPushButton(newWin)
newWin.playButton.setIcon(QtGui.QIcon('run1.png'))
newWin.connect(newWin.playButton,QtCore.SIGNAL('clicked()'),playMovie)


newWin.slider = QtGui.QSlider(QtCore.Qt.Horizontal,newWin.centralwidget)
newWin.verticalLayout.addWidget(newWin.slider)
newWin.verticalLayout.addWidget(newWin.playButton)


newWin.slider.setRange(0,len(q)-2)
newWin.slider.setTickPosition(0)
newWin.connect(newWin.slider, QtCore.SIGNAL('valueChanged(int)'), getValue)

newWin.setWindowState(Qt.WindowMaximized)
w =  newWin.mgl
cnf = q[len(q)-1]
for i in range(len(cnf)):
    a = locals()[cnf[i][0]](w,cnf[i][1],cnf[i][2])
    a.setCellParentProps(cnf[i][3],cnf[i][4],1,0,0)
    w.vizObjects.append(a)
w.setColorMap()

#w.translate([0,-3,-75])
#w.rotate([1,0,0],-50)

w.updateGL()

         
sys.exit(app.exec_())
