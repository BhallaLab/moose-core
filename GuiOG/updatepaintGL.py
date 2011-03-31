import sys
from PyQt4 import QtCore, QtGui
from PyGLWidget import PyGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from oglfunc.objects import *
from oglfunc.group import *

import moose
mc=moose.context

class updatepaintGL(PyGLWidget):
	
    def paintGL(self):
        PyGLWidget.paintGL(self)
	self.render()

    def repaintGL(self):	#update the canvas with coordinates
	mode =1
	
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

	self.selectedObjects.render()		

	
    def drawNewCell(self, cellName, style = 2, cellCentre=[0.0,0.0,0.0], cellAngle=[0.0,0.0,0.0,0.0]):		#cellName = moosepath
    	if style ==1:
    		self.lights = 0
    	else: 
    		self.lights = 1
    		
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

	if (self.selectionMode):		#cells are pickable
		newCell = cellStruct(self,l_coords,cellName,style)
		newCell._centralPos = cellCentre
		newCell.rotation = cellAngle
		self.sceneObjectNames.append(cellName)
		self.sceneObjects.append(newCell)	
	else:					#comapartments are pickable
		for i in range(0,len(l_coords),1):
			if (moose.Compartment(ch[i]).name=='soma'):
				compartmentLine = somaSphere(self,l_coords[i],cellName)
			else:
				if style==1:
					compartmentLine=cLine(self,l_coords[i],cellName)
				else: 	#style==2
					compartmentLine=cCylinder(self,l_coords[i],cellName)
			compartmentLine._centralPos = cellCentre
			compartmentLine.rotation = cellAngle
			self.sceneObjectNames.append(l_coords[i][7])
	    		self.sceneObjects.append(compartmentLine)	

    def get_childrenOfField(self,all_ch,field):	#'all_ch' is a tuple of moose.id, 'field' is the field to sort with; returns a tuple with valid moose.id's
        ch=[]
        for i in range(0,len(all_ch)):	
	    if(mc.className(all_ch[i])==field):
	        ch.append(all_ch[i])
        return tuple(ch)  
	
