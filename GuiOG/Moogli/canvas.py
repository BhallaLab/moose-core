#Author:Chaitanya CH
#FileName: canvas.py

#This program is free software; you can redistribute it and/or
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

import sys
import os
from PyQt4 import QtCore, QtGui

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from openGLHeaders.PyGLWidget import PyGLWidget
from openGLHeaders.objects import *
from openGLHeaders.group import *

from numpy import arange,digitize
from defaults import *
import pickle

class canvas(PyGLWidget):
	
    def initializeGL(self):
        # OpenGL state
        #backGroundColor
        glClearColor(DEFAULT_BGCOLOR[0],DEFAULT_BGCOLOR[1],DEFAULT_BGCOLOR[2],DEFAULT_BGCOLOR[3]) 
        glEnable(GL_DEPTH_TEST)
        self.reset_view()

    def paintGL(self):
        PyGLWidget.paintGL(self)
	self.render()

    def clearCanvas(self):
        self.selectedObjects.removeAll()
        self.vizObjects = {}
        self.sceneObjects = {}
        self.cellComptDict = {}

    def render(self):
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)

        light0_pos = DEFAULT_LIGHT_POSITION
        diffuse0 = DEFAULT_DIFFUSE_COLOR 
        specular0 =DEFAULT_SPECULAR_COLOR
        ambient0 = DEFAULT_AMBIENT_COLOR

        glMatrixMode(GL_MODELVIEW)
        glLightfv(GL_LIGHT0, GL_POSITION, light0_pos)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse0)
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular0)
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient0)

	for obj in self.sceneObjects.values():
	    obj.render() #rendering all the compartments
	    
	for obj in self.vizObjects.values():
	    obj.render() #rendering the color changed compartments, rendering twice! ~optimize!

	self.renderAxis()	#draws axes at right corner
	self.selectedObjects.render()	
	
    def addToVisualize(self,sceneObjectName):
        if sceneObjectName in self.sceneObjects:
            self.vizObjects[sceneObjectName] = self.sceneObjects[sceneObjectName]
        else:
            'This compartment has not been drawn on canvas'

    # def updateViz(self):
    # 	if self.gridRadiusViz==0:
    #     	vals=[]
    #     	for name in self.vizObjectNames:
    #     		r=mc.pathToId(name+self.moosepath)
    #     		d=float(mc.getField(r,self.variable))
    #                     vals.append(d)
    #     	inds = digitize(vals,self.stepVals)

    #     	for i in range(0,len(self.vizObjects)):
    #     		self.vizObjects[i].r,self.vizObjects[i].g,self.vizObjects[i].b=self.colorMap[inds[i]-1]

    #     else:
    #     	vals=[]
    #     	vals_2=[]
    #     	for name in self.vizObjectNames:
    #                 r=mc.pathToId(name+self.moosepath)
    #                 d=float(mc.getField(r,self.variable))
                    

    #                 r2=mc.pathToId(name+self.moosepath_2)
    #                 d2=float(mc.getField(r2,self.variable_2))
				
    #                 vals.append(d)
    #                 vals_2.append(d2)
			
    #     	inds = digitize(vals,self.stepVals)
    #     	inds_2 = digitize(vals_2,self.stepVals_2)

    #     	for i in range(0,len(self.vizObjects)):
    #     		self.vizObjects[i].r,self.vizObjects[i].g,self.vizObjects[i].b=self.colorMap[inds[i]-1]
    #     		self.vizObjects[i].radius=self.indRadius[inds_2[i]-1]

    #     self.updateGL()
    
    def drawNewCompartment(self,cellName,name,coords,style=3,cellCentre=[0.0,0.0,0.0],cellAngle=[0.0,0.0,0.0,0.0]):
        ''' name = 'segmentName',cellName= 'mitral', coords = [x0,y0,z0,x,y,z,d] , style = 1/2/3/4'''
        if (coords[0]!=coords[3] or coords[1]!=coords[4] or coords[2]!=coords[5]): #not a soma
            if style == 1 : #disk
                compartment = somaDisk(self,name,cellName,coords,cellCentre,cellAngle)
            elif style == 2: #line
                compartment = cLine(self,name,cellName,coords,cellCentre,cellAngle)
            elif style == 3: #cylinder
                compartment = cCylinder(self,name,cellName,coords,cellCentre,cellAngle)
            elif style == 4: #capsule
                compartment = cCapsule(self,name,cellName,coords,cellCentre,cellAngle)
        else: #soma case
            if style == 1 : #disk
                compartment = somaDisk(self,name,cellName,coords,cellCentre,cellAngle)
            else: #sphere
                compartment = somaSphere(self,name,cellName,coords,cellCentre,cellAngle)

        self.sceneObjects[cellName+'/'+name] = compartment #add cmpt to sceneobjects

        if self.cellComptDict.has_key(cellName): #add this segment to the cell-compt dict
            self.cellComptDict[cellName].append(compartment)
        else:
            self.cellComptDict[cellName] = [compartment]


    def copyCell(self,newCellName,oldCellName,newCellCentre=[0.0,0.0,0.0],newCellAngle=[0.0,0.0,0.0,0.0]):	
        if self.cellComptDict.has_key(oldCellName):
            for cmpt in self.cellComptDict[oldCellName]:

                newCmpt = cmpt
                newCmpt.setCentralPosition(newCellCentre)
                newCmpt.setRotation(newCellAngle)

                newCmptName = '/'+newCellName+'/'+newCmpt.name
                self.sceneObjects[newCmptName] = newCmpt

                if self.cellComptDict.has_key(newCellName):
                    self.cellComptDict[newCellName].append(newCmpt)
                else:
                    self.cellComptDict[newCellName] = [newCmpt]

        else:
            print 'No cell named ',oldCellName,' previously drawn.'

           
class colorMap(object):
    def __init__(self,fileName,minVal,maxVal,label):
        f = open(os.path.join(PATH_COLORMAPS,fileName),'r')
        self.colorMap = pickle.load(f)
        f.close()
        self.fileName = fileName
        self.steps = len(self.colorMap)
        self.label = label
        self.minVal = minVal
        self.maxVal = maxVal
        self.stepVals = arange(self.minVal,self.maxVal,(self.maxVal-self.minVal)/self.steps)
    
    def setMinMaxValue(self,minVal,maxVal):
        self.minVal = minVal
        self.maxVal = maxVal
        self.stepVals = arange(self.minVal,self.maxVal,(self.maxVal-self.minVal)/self.steps)

    def setColorMap(self,fileName,label,minVal=None,maxVal=None):
        f = open(os.path.join(PATH_COLORMAPS,fileName),'r')
        self.colorMap = pickle.load(f)
        f.close()
        self.fileName = fileName
        self.label = label
        self.steps = len(self.colorMap)
        if (minVal != None):
            if (maxVal != None):
                self.setMinMaxValue(minVal,maxVal)


class newGLWindow(QtGui.QMainWindow):
    def __init__(self, parent = None):
        # initialization of the superclass
        super(newGLWindow, self).__init__(parent)
        # setup the GUI --> function generated by pyuic4
        self.name = 'GL Window '
        #self.setupUi(self)

    def windowTitle(self,name):
    	self.name = name
    	self.setupUi(self)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(500, 500)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
	self.canvas = canvas(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.canvas.sizePolicy().hasHeightForWidth())
        self.canvas.setSizePolicy(sizePolicy)
        self.canvas.setObjectName("canvas")
        self.verticalLayout.addWidget(self.canvas)
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", self.name, None, QtGui.QApplication.UnicodeUTF8))

