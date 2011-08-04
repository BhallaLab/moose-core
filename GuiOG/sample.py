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
    print 'playing movie'
    t = newWin.slider.tickPosition()
    newWin.playButton.setEnabled(False)
    while t<(len(q)-1):
#        newWin.slider.setTickPosition(t)
        time.sleep(0.2)
        updateColor(w,q,t)
        t += 1

    newWin.playButton.setEnabled(True)


app = QtGui.QApplication(sys.argv) 
newWin = newGLWindow()
newWin.show()
newWin.setWindowState(Qt.WindowMaximized)

f = open('tada.pkl','r')
q = pickle.load(f)
f.close()


newWin.playButton =  QtGui.QToolButton(newWin)
newWin.playButton.setText('>')
newWin.connect(newWin.playButton,QtCore.SIGNAL('clicked()'),playMovie)


newWin.slider = QtGui.QSlider(QtCore.Qt.Horizontal,newWin.centralwidget)
newWin.verticalLayout.addWidget(newWin.playButton)
#newWin.horizontalLayout.addWidget(newWin.slider)

newWin.verticalLayout.addWidget(newWin.slider)
newWin.slider.setRange(0,len(q)-2)
newWin.slider.setTickPosition(0)
newWin.connect(newWin.slider, QtCore.SIGNAL('valueChanged(int)'), getValue)

#newWin.setWindowState(Qt.WindowMaximized)
w =  newWin.mgl
cnf = q[len(q)-1]
for i in range(len(cnf)):
    a = locals()[cnf[i][0]](w,cnf[i][1],cnf[i][2])
    a.setCellParentProps(cnf[i][3],cnf[i][4],1,0,0)
    w.vizObjects.append(a)
w.setColorMap()
w.updateGL()

    

         
sys.exit(app.exec_())
