#Author:Chaitanya CH
#FileName: updatepaintGL.py

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
from PyQt4 import QtCore, QtGui
from PyGLWidget import PyGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from oglfunc.objects import *
from oglfunc.group import *

from numpy import arange,digitize
import moose
import pickle
mc=moose.context

class updatepaintGL(PyGLWidget):
	
    def paintGL(self):
        PyGLWidget.paintGL(self)
	self.render()

    def setSelectionMode(self,mode):	
	self.selectionMode = mode	
	
    def render(self):
	if self.lights:
		glMatrixMode(GL_MODELVIEW)
		glEnable(GL_LIGHTING)
		glEnable(GL_LIGHT0)
		glEnable(GL_COLOR_MATERIAL)

		light0_pos = 200.0, 200.0, 300.0, 0
		diffuse0 = 1.0, 1.0, 1.0, 1.0
		specular0 = 1.0, 1.0, 1.0, 1.0
		ambient0 = 0, 0, 0, 1

		glMatrixMode(GL_MODELVIEW)
		glLightfv(GL_LIGHT0, GL_POSITION, light0_pos)
		glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse0)
		glLightfv(GL_LIGHT0, GL_SPECULAR, specular0)
		glLightfv(GL_LIGHT0, GL_AMBIENT, ambient0)
	self.renderAxis()	#draws 3 axes at origin
	
	for obj in self.sceneObjects:
	    obj.render()
	    
	for obj in self.vizObjects:
	    obj.render()
	
	self.selectedObjects.render()	
	
    def updateViz(self):
    	if self.gridRadiusViz==0:
		vals=[]
		for name in self.vizObjectNames:
			r=mc.pathToId(name+self.moosepath)
			d=float(mc.getField(r,self.variable))
                        vals.append(d)
		inds = digitize(vals,self.stepVals)

		for i in range(0,len(self.vizObjects)):
			self.vizObjects[i].r,self.vizObjects[i].g,self.vizObjects[i].b=self.colorMap[inds[i]-1]

	else:
		vals=[]
		vals_2=[]
		for name in self.vizObjectNames:
                    r=mc.pathToId(name+self.moosepath)
                    d=float(mc.getField(r,self.variable))
                    

                    r2=mc.pathToId(name+self.moosepath_2)
                    d2=float(mc.getField(r2,self.variable_2))
				
                    vals.append(d)
                    vals_2.append(d2)
			
		inds = digitize(vals,self.stepVals)
		inds_2 = digitize(vals_2,self.stepVals_2)

		for i in range(0,len(self.vizObjects)):
			self.vizObjects[i].r,self.vizObjects[i].g,self.vizObjects[i].b=self.colorMap[inds[i]-1]
			self.vizObjects[i].radius=self.indRadius[inds_2[i]-1]

	self.updateGL()
    
    def setSpecificCompartmentName(self,name):
    	self.specificCompartmentName = name
			
    def drawNewCell(self, cellName, style = 2,cellCentre=[0.0,0.0,0.0],cellAngle=[0.0,0.0,0.0,0.0]):	
    	#***cellName = moosepath in the GL canvas***
	an=moose.Neutral(cellName)
	all_ch=an.childList 					#all children
	ch = self.get_childrenOfField(all_ch,'Compartment')	#compartments only
	l_coords = []
	for i in range(0,len(ch),1):
    	    	x=float(mc.getField(ch[i],'x'))*(1e+04)
    	    	y=float(mc.getField(ch[i],'y'))*(1e+04)
    	    	z=float(mc.getField(ch[i],'z'))*(1e+04)
    	    	x0=float(mc.getField(ch[i],'x0'))*(1e+04)
    	    	y0=float(mc.getField(ch[i],'y0'))*(1e+04)
	   	z0=float(mc.getField(ch[i],'z0'))*(1e+04)
	   	d=float(mc.getField(ch[i],'diameter'))*(1e+04)
    	    	l_coords.append((x0,y0,z0,x,y,z,d,ch[i].path()))
    	    	
    	if self.viz==1:				#fix
    		self.selectionMode=0
    		
    	if (style==1) or (style==2):		#ensures soma is drawn as a sphere
    		self.specificCompartmentName='soma'

	if (self.selectionMode):		#self.selectionMode = 1,cells are pickable
		newCell = cellStruct(self,l_coords,cellName,style,specificCompartmentName=self.specificCompartmentName)
		newCell._centralPos = cellCentre
		newCell.rotation = cellAngle
		self.sceneObjectNames.append(cellName)
		self.sceneObjects.append(newCell)	
		if self.viz==1:			#fix
			self.vizObjects.append(newCell)
			self.vizObjectNames.append(cellName)
			
	else:					#self.selectionMode=0,comapartments are pickable
		for i in range(0,len(l_coords),1):
			if (moose.Compartment(ch[i]).name==self.specificCompartmentName):#drawing of the select compartment in style 0
				if style==0:
					compartmentLine=somaDisk(self,l_coords[i],cellName)
					compartmentLine._centralPos = cellCentre
					compartmentLine.rotation = cellAngle
					self.sceneObjectNames.append(l_coords[i][7])
	    				self.sceneObjects.append(compartmentLine)
	    		
	    				if self.viz==1:
						self.vizObjects.append(compartmentLine)
						self.vizObjectNames.append(l_coords[i][7])
				
				elif (style==1) or (style==2):				#drawing of the soma in style 1&2
					compartmentLine = somaSphere(self,l_coords[i],cellName) 	#$
					
				elif style==3:						#grid view, any choice to compartment as a disk
					compartmentLine=somaDisk(self,[0,0,0,0,0,0,0,l_coords[i][7]],cellName)
					compartmentLine.radius = 0.20
					compartmentLine._centralPos = cellCentre
					compartmentLine.rotation = cellAngle
					self.sceneObjectNames.append(l_coords[i][7])
	    				self.sceneObjects.append(compartmentLine)
	    		
	    				if self.viz==1:
						self.vizObjects.append(compartmentLine)
						self.vizObjectNames.append(l_coords[i][7])
				
			else:								#to draw compartments other than soma
				if style==1:
					compartmentLine=cLine(self,l_coords[i],cellName)	#$
					
				elif style==2:
					compartmentLine=cCylinder(self,l_coords[i],cellName)	#$
			
			
			if (style==1)or(style==2):					#necessary, appends includ the soma as well (= include code in "$" areas)
				compartmentLine._centralPos = cellCentre
				compartmentLine.rotation = cellAngle
				self.sceneObjectNames.append(l_coords[i][7])
	    			self.sceneObjects.append(compartmentLine)
	    		
	    			if self.viz==1:
					self.vizObjects.append(compartmentLine)
					self.vizObjectNames.append(l_coords[i][7])	    		

    def drawAllCells(self, style = 2, cellCentre=[0.0,0.0,0.0], cellAngle=[0.0,0.0,0.0,0.0]):
        an=moose.Neutral('/')						#moose root children
	all_ch=an.childList 	
						#all children under root, of cell type
	ch = self.get_childrenOfField(all_ch,'Cell')
	for i in range(0,len(ch),1):
	    self.drawNewCell(moose.Cell(ch[i]).path,style,cellCentre,cellAngle)
	    
	nh = self.get_childrenOfField(all_ch,'Neutral')			#all cells under all other neutral elements.	
	for j in range(0,len(nh),1):
	    an=moose.Neutral(nh[j])					#this neutral element
	    all_ch=an.childList 					#all children under this neutral element
	    ch = self.get_childrenOfField(all_ch,'Cell')
	    for i in range(0,len(ch),1):
	    	self.drawNewCell(moose.Cell(ch[i]).path,style,cellCentre,cellAngle)
	    	
	
    def drawAllCellsUnder(self, path, style = 2, cellCentre=[0.0,0.0,0.0], cellAngle=[0.0,0.0,0.0,0.0]):
	    pathID = mc.pathToId(path)
	    if mc.className(pathID) =='Neutral':
		an=moose.Neutral(pathID)					#this neutral element
	    	all_ch=an.childList 					#all children under this neutral element
	    	ch = self.get_childrenOfField(all_ch,'Cell')
	    	for i in range(0,len(ch),1):
	    		self.drawNewCell(moose.Cell(ch[i]).path,style,cellCentre,cellAngle)	 
	    else: 	
	    	print 'Select a Neutral element path'   	
	    

    def get_childrenOfField(self,all_ch,field):	#'all_ch' is a tuple of moose.id, 'field' is the field to sort with; returns a tuple with valid moose.id's
        ch=[]
        for i in range(0,len(all_ch)):	
	    if(mc.className(all_ch[i])==field):
	        ch.append(all_ch[i])
        return tuple(ch)  
        
    def setColorMap(self,vizMinVal=-0.1,vizMaxVal=0.07,moosepath='',variable='Vm',cMap='jet'):
    	self.colorMap=[]
    	self.stepVals=[]
    	self.moosepath=moosepath
    	self.variable=variable
    	if cMap=='':
    		steps = 64
    		for x in range(0,steps):
	 		r=max((2.0*x)/steps-1,0.0)
			b=max((-2.0*x)/steps+1,0.0)
			g=min((2.0*x)/steps,(-2.0*x)/steps+2)
			self.colorMap.append([r,g,b])
	else:
		f = open(cMap,'r')
		self.colorMap = pickle.load(f)
		steps = len(self.colorMap)
		f.close()
	self.stepVals = arange(vizMinVal,vizMaxVal,(vizMaxVal-vizMinVal)/steps)
	
    def setColorMap_2(self,vizMinVal_2=-0.1,vizMaxVal_2=0.07,moosepath_2='',variable_2='Vm'):	#colormap for the radius - grid view case
    	self.moosepath_2 = moosepath_2
    	self.variable_2 = variable_2
    	self.stepVals_2 = arange(vizMinVal_2,vizMaxVal_2,(vizMaxVal_2-vizMinVal_2)/30)		#assigned a default of 30 steps
    	self.indRadius = arange(0.05,0.20,0.005)						#radius equivalent colormap 
	
class newGLWindow(QtGui.QMainWindow):
    def __init__(self, parent = None):
        # initialization of the superclass
        super(newGLWindow, self).__init__(parent)
        # setup the GUI --> function generated by pyuic4
        self.name = 'GL Window '
        #self.setupUi(self)

    def windowTitle(self,name):
    	self.name = name
    	#MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", self.name, None, QtGui.QApplication.UnicodeUTF8))
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
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
	self.mgl = updatepaintGL(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mgl.sizePolicy().hasHeightForWidth())
        self.mgl.setSizePolicy(sizePolicy)
        self.mgl.setObjectName("mgl")
        self.horizontalLayout.addWidget(self.mgl)
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", self.name, None, QtGui.QApplication.UnicodeUTF8))



class newGLSubWindow(QtGui.QMdiSubWindow):
    """This is to customize MDI sub window for our purpose.

    In particular, we don't want anything to be deleted when the window is closed. 
    
    """
    def __init__(self, *args):
        QtGui.QMdiSubWindow.__init__(self, *args)
        
    def closeEvent(self, event):
        self.emit(QtCore.SIGNAL('subWindowClosed()'))
        self.hide()

