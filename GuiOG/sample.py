import sys
from PyQt4 import QtGui,QtCore
from PyQt4.Qt import Qt
import pickle
from header import newGLWindow
from oglfunc.sobjects import *

playTimeStep = 200

def updateColor(w,q,t):
    for i in range(len(w.vizObjects)):
        w.vizObjects[i].r,w.vizObjects[i].g,w.vizObjects[i].b = w.colorMap[q[t][i]-1]
    w.updateGL()

def saveAsMovie(w,startFrame,stopFrame):
    for framenum in range(startFrame,stopFrame+1):
        newWin.slider.setValue(framenum)
        pic = w.grabFrameBuffer()
        pic.save('movie/sim_'+str(framenum)+'.png','PNG')
    
    tempPath = os.getcwd()
    os.chdir('movie')
    f = open('filelist.txt','w')
    filelist = ["sim_"+str(i)+".png" for i in range(startFrame,stopFrame+1)]
    f.write('\n'.join(filelist))
    f.close()

    ## mpeg4 compression
    os.system('mencoder mf://@filelist.txt -mf w=1280:h=800:fps=10:type=png '\
                  ' -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o simulation.avi')

    os.chdir(tempPath)

def saveFrames():
    w.inputDialog = QtGui.QInputDialog()
    text,pressOK =  w.inputDialog.getText(w.inputDialog,'Save Frames(Movie)','First,Last:')
    if pressOK:
        frameNumbers = str(text).split(',')
        if len(frameNumbers) == 2 :
            if (int(frameNumbers[0]) < int(frameNumbers[1])):
                saveAsMovie(w,int(frameNumbers[0]),int(frameNumbers[1]))
            else:
                print 'starting frame > ending frame, encountered a causality Error'
        else:
            print 'wrong inputs'

def getValue(value):
    updateColor(w,q,value)
    statusBar.showMessage(str('Showing '+ str(value)+ '/' +str(len(q)-2)))
    
def playMovie():
    t = newWin.slider.value()
    t += 1        
    if t < len(q)-1:
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
    newWin.ctimer.start(playTimeStep)

def stopTimer():
    newWin.ctimer.stop()


app = QtGui.QApplication(sys.argv) 
newWin = newGLWindow()
newWin.show()
newWin.setWindowState(Qt.WindowMaximized)

f = open(sys.argv[1],'r')
q = pickle.load(f)
f.close()

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

#saveButton
newWin.saveButton = QtGui.QToolButton(newWin.toolbar)
newWin.saveButton.setIcon(QtGui.QIcon('save.png'))
newWin.saveButton.setGeometry(60,0,30,30)
newWin.connect(newWin.saveButton, QtCore.SIGNAL('clicked()'),saveFrames)

#slider
newWin.slider = QtGui.QSlider(QtCore.Qt.Horizontal,newWin.centralwidget)
newWin.slider.setRange(0,len(q)-2)
newWin.slider.setTickPosition(0)
newWin.connect(newWin.slider, QtCore.SIGNAL('valueChanged(int)'), getValue)

#statusbar
statusBar = QtGui.QStatusBar(newWin)
newWin.setStatusBar(statusBar)
statusBar.showMessage(str('Loaded ' + sys.argv[1]+ ', has '+ str(len(q)-2)+ ' time instances'))

#default setting
#newWin.verticalLayout.addWidget(newWin.toolbar)
newWin.verticalLayout.addWidget(newWin.slider)
newWin.setWindowState(Qt.WindowMaximized)

w =  newWin.mgl
cnf = q[len(q)-1]
for i in range(len(cnf)):
    a = locals()[cnf[i][0]](w,cnf[i][1],cnf[i][2])
    a.setCellParentProps(cnf[i][3],cnf[i][4],1,0,0)
    w.vizObjects.append(a)
w.setColorMap()
w.updateGL()
         
sys.exit(app.exec_())
