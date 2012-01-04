from numpy import genfromtxt,array 
import os
import bz2
import pickle
import moose
import sys
from PyQt4 import QtGui,QtCore
from PyQt4.Qt import Qt
from StringIO import StringIO
from updatepaintGL import *

#movie saving defaults
movie_fps = 10
movie_width = 1280
movie_height = 800
movie_filename = 'simulation.avi' 

#visualization mode, visualize single parameter (=1); to visualize 2 simultaneously (=2)
numberOfParaToViz = 2  #(= 2 is Visualize_MAPKglu / two parameters simultaneously) (=1 is to visualize just Ca)
parametersToViz = ['MAPK','glu'] #applicable only when numberOfParaToViz = 2, 1 name gets colormap, 2 gets colormap2

#lump frames for visualization
lumpWindowWidth = 50  #number of frames to lump together
lumpWindowHow = 2 #no lumping = 0 #mean = 1 #max = 2 

#cell file name, location
cellFileName = 'display_ca1_v1.p'
pathToCellFile = os.getcwd()+'/'+cellFileName

#simulation folder
folder = 'E1_0.0005_5_0.5_v3.output'

#subfolder insider simulation folder
subFolder = 'MAPK_glu'
pathToData = os.getcwd()+'/'+folder+'/'+subFolder

#Min-Max values - defaults
Ca_minVal = 0.05
Ca_maxVal = 1.0
MAPK_minVal = 0
MAPK_maxVal = 0.1
glu_minVal = 0
glu_maxVal = 4.0e-12

#colormaps
cMap = os.getcwd()+'/oglfunc/colors/'+'jet' 
cMap2 = os.getcwd()+'/oglfunc/colors/'+'jet'

#update time in ms - for the player
updateTime = 40

#reading data.
data = [0]*len(os.listdir(pathToData)) #empty array to dump data.
compt_names = []
para_names = []

for i,f in enumerate(os.listdir(pathToData)):
    data[i] = numpy.genfromtxt(StringIO(bz2.BZ2File(pathToData+'/'+f).read()))
    compt_names.append(f.split('_')[1].split('.')[0]) #to get the ordering of the data when visualizing.
    para_names.append(f.split('_')[0]) #parameter being viz

numberOfCompartments = len(data)
timeInstances = len(data[0])

#lumping
if lumpWindowHow != 0:
    for j in range(len(data)):
        lumpData =[]
        for i in range(0,timeInstances,lumpWindowWidth):
            if lumpWindowHow == 1:
                lumpData.append(array(data[j][i:min((i+lumpWindowWidth),timeInstances)]).mean())
            else:
#                print i,data[i:min((i+lumpWindowWidth),timeInstances)]
#                print array(data[i:min((i+lumpWindowWidth),timeInstances)]).max()
                lumpData.append(array(data[j][i:min((i+lumpWindowWidth),timeInstances)]).max())
        data[j] = lumpData


timeData = []
#restructure data
for i in range(len(data[0])):
    dummy = []
    for j in range(numberOfCompartments):
            dummy.append(data[j][i])
    timeData.append(dummy)

#player window
app = QtGui.QApplication(sys.argv) 
newWin = newGLWindow()
newWin.show()
newWin.setWindowState(Qt.WindowMaximized)
w =  newWin.mgl #canvas 

paraWiseViz = []

#moose reading cell, and plotting in canvas
moose.context.readCell(pathToCellFile,'cell')
w.drawNewCell('/cell',style=2)
for j,name in enumerate(compt_names): #adding those with viz data
    for i,ObjectName in enumerate(w.sceneObjectNames):
        if ObjectName == ('/cell/'+name): 
            w.vizObjects.append(w.sceneObjects[i])
            w.vizObjectNames.append(ObjectName)
            paraWiseViz.append(para_names[j])

w.updateGL()

#stepVals
if numberOfParaToViz == 1:
    f = open(cMap,'r')
    colorMap = pickle.load(f)
    steps = len(colorMap)
    f.close()

    stepVals = arange(Ca_minVal,Ca_maxVal,(Ca_maxVal-Ca_minVal)/steps)

else:
    f = open(cMap,'r')
    colorMap1 = pickle.load(f)
    steps1 = len(colorMap1)
    f.close()
    
    f = open(cMap2,'r')
    colorMap2 = pickle.load(f)
    steps2 = len(colorMap2)
    f.close()

    stepVals1 = arange(MAPK_minVal,MAPK_maxVal,(MAPK_maxVal-MAPK_minVal)/steps1)
    stepVals2 = arange(glu_minVal,glu_maxVal,(glu_maxVal-glu_minVal)/steps2)


#digitize all values before hand saves viz time
timeDataDig = []
firstParaIndex = []
secParaIndex = []

if numberOfParaToViz == 1:
    for aData in timeData:
        timeDataDig.append(digitize(aData,stepVals))
else:
    for i,name in enumerate(para_names):
            if para_names[i] == parametersToViz[0]:
                firstParaIndex.append(i)
            else:
                secParaIndex.append(i)
    for aData in timeData:
        aDatadummy  = array(aData)
        aDatadummy[firstParaIndex] = digitize(aDatadummy[firstParaIndex],stepVals1)
        aDatadummy[secParaIndex] = digitize(aDatadummy[secParaIndex],stepVals2)
        aDatadummy2  = [x.__int__() for x in aDatadummy]
        timeDataDig.append(aDatadummy2)

#player functionz

def updateColor(w,timeDataDig,t):
    if numberOfParaToViz == 1:
        for i in range(len(w.vizObjects)):
            w.vizObjects[i].r,w.vizObjects[i].g,w.vizObjects[i].b = colorMap[timeDataDig[t][i]-1]
        w.updateGL()
    else:
        for i in range(len(w.vizObjects)):
            if paraWiseViz[i] == parametersToViz[0]:
                w.vizObjects[i].r,w.vizObjects[i].g,w.vizObjects[i].b = colorMap1[timeDataDig[t][i]-1]
            else:
                w.vizObjects[i].r,w.vizObjects[i].g,w.vizObjects[i].b = colorMap2[timeDataDig[t][i]-1]
        w.updateGL()

def saveAsMovie(w,startFrame,stopFrame):
    if not os.path.exists('movieMake'):
        os.mkdir('movieMake')

    for framenum in range(startFrame,stopFrame+1):
        print framenum
        newWin.slider.setValue(framenum)
        pic = w.grabFrameBuffer()
        pic.save('movieMake/sim_'+str(framenum)+'.png','PNG')
    
    os.chdir('movieMake')
    f = open('filelist.txt','w')
    filelist = ["sim_"+str(i)+".png" for i in range(startFrame,stopFrame+1)]
    f.write('\n'.join(filelist))
    f.close()

    ## mpeg4 compression
    mpegString = 'mencoder mf://@filelist.txt -mf w='+str(movie_width)+':h='+str(movie_height)+':fps='+str(movie_fps)+':type=png  -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o '+ movie_filename
    os.system(mpegString)
    os.chdir('..')

def getValue(value):
    updateColor(w,timeDataDig,value)
    statusBar.showMessage(str('Showing '+ str(value)+ '/' +str(len(timeData)-1)))

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
    
def playMovie():
    t = newWin.slider.value()
    t += 1        
    if t < len(timeData)-1:
        newWin.slider.setValue(t)
    else:
        stopTimer()
        
def startTimer():
    newWin.ctimer.start(updateTime)

def stopTimer():
    newWin.ctimer.stop()


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
newWin.slider.setRange(0,len(timeData)-1)
newWin.slider.setTickPosition(0)
newWin.connect(newWin.slider, QtCore.SIGNAL('valueChanged(int)'), getValue)

#statusbar
statusBar = QtGui.QStatusBar(newWin)
newWin.setStatusBar(statusBar)
statusBar.showMessage(str('Loaded has '+ str(len(timeData)-1)+ ' time instances'))

#default setting
#newWin.verticalLayout.addWidget(newWin.toolbar)
newWin.verticalLayout.addWidget(newWin.slider)
newWin.setWindowState(Qt.WindowMaximized)

sys.exit(app.exec_())


