import sys
from PyQt4 import QtCore, QtGui
from PyGLWidget import PyGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from oglfunc.objects import *
from oglfunc.group import *


class updatepaintGL(PyGLWidget):
    def paintGL(self):
        PyGLWidget.paintGL(self)
	self.render()

    def repaintGL(self,coords):	#update the cell diagram with coordinates
	self.l_coords = coords
	for i in range(0,len(self.l_coords),1):	
	    newline=cLine(self,self.l_coords[i])
	    self.sceneObjects.append(newline)	 

    def render(self):
	for obj in self.sceneObjects:
	    obj.render()
	self.selectedObjects.render()		


    #def seltool(self):
#	self.toolsel = 1	

#    def mousePressEvent(self, _event):
#	if (self.toolsel==1):
	   # if (_event.button()==1):
	   # 	self.selectiontool(_event.pos())
		#self.current_point_2D_ = _event.pos()            	
		#self.current_point_ok_, self.current_point_3D_ = self.map_to_sphere(self.current_point_2D_)
	    	#self.selectiontool(self.current_point_3D_)		

	
