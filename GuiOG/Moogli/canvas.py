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

import OpenGL.GL as ogl
import OpenGL.GLUT as oglut #using this only for denoting x,y,z on axis -> reconsider!

from openGLHeaders.objects import *
from openGLHeaders.group import *
from PyQGLViewer import *

from numpy import arange,digitize
from defaults import *
import pickle

class canvas(QGLViewer):
    compartmentSelected = QtCore.pyqtSignal(QtCore.QString)
    def __init__(self,parent=None):
        QGLViewer.__init__(self,parent)
        self.setStateFileName('.MoogliState.xml')

        self.selectedObjects = Group(self)
        self.vizObjects = {}
        self.sceneObjects = {}
        self.cellComptDict = {}

#        self.__rectangle = QtCore.QRect()
#        self.__selectionMode = 0
#        self.__objects = []
#        self.__selection = []

    def init(self):
        self.restoreStateFromFile()
#        self.setBackgroundColor(QtGui.QColor(255,255,255,255))
        self.setSceneRadius(10.0)

    def clearCanvas(self):
        self.selectedObjects.removeAll()
        self.vizObjects = {}
        self.sceneObjects = {}
        self.cellComptDict = {}

    def draw(self):
        for obj in self.sceneObjects.values():
            obj.render()
        for obj in self.vizObjects.values():#rendering the color changed compartments, rendering twice! ~optimize!
            obj.render() 
        self.selectedObjects.render()
 #       if self.__selectionMode != 0:
 #           self.__drawSelectionRectangle()

    def postDraw(self):
        QGLViewer.postDraw(self)
        self.drawCornerAxis()

    def drawWithNames(self):
        for i in range(len(self.sceneObjects.items())):
            ogl.glPushMatrix()
            ogl.glPushName(i)
            self.sceneObjects.items()[i][1].render()
            ogl.glPopName()
            ogl.glPopMatrix()

    def postSelection(self,point):
        if self.selectedName() == -1:
            self.selectedObjects.removeAll()
        else:
            self.compartmentSelected.emit(str(self.sceneObjects.items()[int(self.selectedName())][0]))
            self.selectedObjects.add(self.sceneObjects.items()[int(self.selectedName())][1])

    # def __drawSelectionRectangle(self):
    #     self.startScreenCoordinatesSystem()
    #     ogl.glDisable(ogl.GL_LIGHTING)
    #     ogl.glEnable(ogl.GL_BLEND)

    #     ogl.glColor4f(0.0, 0.0, 0.3, 0.3)
    #     ogl.glBegin(ogl.GL_QUADS)
    #     ogl.glVertex2i(self.__rectangle.left(),  self.__rectangle.top())
    #     ogl.glVertex2i(self.__rectangle.right(), self.__rectangle.top())
    #     ogl.glVertex2i(self.__rectangle.right(), self.__rectangle.bottom())
    #     ogl.glVertex2i(self.__rectangle.left(),  self.__rectangle.bottom())
    #     ogl.glEnd()

    #     ogl.glLineWidth(2.0)
    #     ogl.glColor4f(0.4, 0.4, 0.5, 0.5)
    #     ogl.glBegin(ogl.GL_LINE_LOOP)
    #     ogl.glVertex2i(self.__rectangle.left(),  self.__rectangle.top())
    #     ogl.glVertex2i(self.__rectangle.right(), self.__rectangle.top())
    #     ogl.glVertex2i(self.__rectangle.right(), self.__rectangle.bottom())
    #     ogl.glVertex2i(self.__rectangle.left(),  self.__rectangle.bottom())
    #     ogl.glEnd()

    #     ogl.glDisable(ogl.GL_BLEND)
    #     ogl.glEnable(ogl.GL_LIGHTING)
    #     self.stopScreenCoordinatesSystem()

    # def endSelection(self,p):
    #     selection = self.getMultipleSelection()
    #     for zmin,zmax,id in selection:
    #         if self.__selectionMode == 1:
    #             self.addIdToSelection(id)
    #         elif self.__selectionMode == 2 : 
    #             self.removeIdFromSelection(id)
    #     self.__selectionMode = 0
    
    # def mousePressEvent(self,e):
    #     """ Mouse events functions """
    #     # Start selection. Mode is ADD with Shift key and TOGGLE with Alt key.
    #     self.__rectangle = QtCore.QRect(e.pos(), e.pos())
    #     if e.button() == QtCore.Qt.LeftButton and e.modifiers() == QtCore.Qt.ShiftModifier:
    #         self.__selectionMode = 1
    #     elif e.button() == QtCore.Qt.LeftButton and e.modifiers() == QtCore.Qt.AltModifier:
    #         self.__selectionMode = 2
    #     else:
    #         QGLViewer.mousePressEvent(self,e)
    
    # def mouseMoveEvent(self,e):
    #     if self.__selectionMode != 0:
    #         # Updates rectangle_ coordinates and redraws rectangle
    #         self.__rectangle.setBottomRight(e.pos())
    #         self.updateGL()
    #     else:
    #         QGLViewer.mouseMoveEvent(self,e)
    
    # def mouseReleaseEvent(self,e):
    #     if self.__selectionMode != 0:
    #         # Actual selection on the rectangular area.
    #         # Possibly swap left/right and top/bottom to make rectangle_ valid.
    #         self.__rectangle = self.__rectangle.normalized()
    #         # Define selection window dimensions
    #         self.setSelectRegionWidth(self.__rectangle.width())
    #         self.setSelectRegionHeight(self.__rectangle.height())
    #         # Compute rectangle center and perform selection
    #         self.select(self.__rectangle.center())
    #         # Update display to show new selected objects
    #         self.updateGL()
    #     else:
    #         QGLViewer.mouseReleaseEvent(self,e)

    # def addIdToSelection(self,id):
    #     if not id in self.__selection:
    #         self.__selection.append(id)
    
    # def removeIdFromSelection(self,id):
    #     self.__selection.remove(id)
    
    def addToVisualize(self,sceneObjectName):
        if sceneObjectName in self.sceneObjects:
            self.vizObjects[sceneObjectName] = self.sceneObjects[sceneObjectName]
        else:
            'This compartment has not been drawn on canvas'

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

    def drawCornerAxis(self):
        # The viewport and the scissor are changed to fit the lower left
        # corner. Original values are saved.
        viewport = ogl.glGetIntegerv(ogl.GL_VIEWPORT)
        scissor  = ogl.glGetIntegerv(ogl.GL_SCISSOR_BOX)

        # Axis viewport size, in pixels
        size = 150;
        ogl.glViewport(0,0,size,size)
        ogl.glScissor(0,0,size,size)

        # The Z-buffer is cleared to make the axis appear over the
        # original image.
        ogl.glClear(ogl.GL_DEPTH_BUFFER_BIT)

        # Tune for best line rendering
        ogl.glDisable(ogl.GL_LIGHTING)
        ogl.glLineWidth(3.0)

        ogl.glMatrixMode(ogl.GL_PROJECTION)
        ogl.glPushMatrix()
        ogl.glLoadIdentity()
        ogl.glOrtho(-1, 1, -1, 1, -1, 1)

        ogl.glMatrixMode(ogl.GL_MODELVIEW)
        ogl.glPushMatrix()
        ogl.glLoadIdentity()
        ogl.glMultMatrixd(self.camera().orientation().inverse().matrix())

        ogl.glBegin(ogl.GL_LINES)
        ogl.glColor3f(1.0, 0.0, 0.0)
        ogl.glVertex3f(0.0, 0.0, 0.0)
        ogl.glVertex3f(0.5, 0.0, 0.0)

        ogl.glColor3f(0.0, 1.0, 0.0)
        ogl.glVertex3f(0.0, 0.0, 0.0)
        ogl.glVertex3f(0.0, 0.5, 0.0)
    
        ogl.glColor3f(0.0, 0.0, 1.0)
        ogl.glVertex3f(0.0, 0.0, 0.0)
        ogl.glVertex3f(0.0, 0.0, 0.5)
        ogl.glEnd()

        oglut.glutInit()
        ogl.glColor(1,0,0)
        ogl.glRasterPos3f(0.6, 0.0, 0.0)
        oglut.glutBitmapCharacter(GLUT_BITMAP_8_BY_13,88)#ascii x
        ogl.glColor(0,1,0)
        ogl.glRasterPos3f(0.0, 0.6, 0.0)
        oglut.glutBitmapCharacter(GLUT_BITMAP_8_BY_13,89)#ascii y
        ogl.glColor(0,0,1)
        ogl.glRasterPos3f(0.0, 0.0, 0.6)
        oglut.glutBitmapCharacter(GLUT_BITMAP_8_BY_13,90)#ascii z

        ogl.glMatrixMode(ogl.GL_PROJECTION)
        ogl.glPopMatrix()

        ogl.glMatrixMode(ogl.GL_MODELVIEW)
        ogl.glPopMatrix()

        ogl.glEnable(ogl.GL_LIGHTING)

        # The viewport and the scissor are restored.
        ogl.glScissor(*scissor)
        ogl.glViewport(*viewport)


           
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
