# -*- coding: utf-8 -*-
#===============================================================================
#
# PyGLWidget.py
#
# A simple GL Viewer.
#
# Copyright (c) 2011, Arne Schmitz <arne.schmitz@gmx.net>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#===============================================================================

from PyQt4 import QtCore, QtGui, QtOpenGL
import math
import numpy
import numpy.linalg as linalg
import OpenGL
OpenGL.ERROR_CHECKING = True
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from group import *

class PyGLWidget(QtOpenGL.QGLWidget):

    # Qt signals
    signalGLMatrixChanged = QtCore.pyqtSignal()
    rotationBeginEvent = QtCore.pyqtSignal()
    rotationEndEvent = QtCore.pyqtSignal()
	
    compartmentSelected = QtCore.pyqtSignal( QtCore.QString)

    def __init__(self, parent = None):
        format = QtOpenGL.QGLFormat()
        format.setSampleBuffers(True)
        QtOpenGL.QGLWidget.__init__(self, format, parent)
       
        
        self.setMouseTracking(True)
        self.modelview_matrix_  = []
        self.translate_vector_  = [0.0, 0.0, 0.0]
        self.viewport_matrix_   = []
        self.projection_matrix_ = []
        self.near_   = 1.0		#0.1
        self.far_    = 200.0
        self.fovy_   = 4.0
        self.radius_ = 10.0
        self.last_point_2D_ = QtCore.QPoint()
        self.last_point_ok_ = False
        self.last_point_3D_ = [1.0, 0.0, 0.0]
        self.isInRotation_  = False
	
        #additions by chaitanya
	#additions by chaitanya
	self.xpan = 0.0
        self.ypan = 0.0
        self.zpan = 0.0

        self.cellComptDict = {}
	#self.lights = 1			#lights	
	self.ctrlPressed = False 	#default no control pressed
        #self.shiftPressed = False       #default no shift pressed
	self.selectedObjects =Group(self)		#each line is a scene object.
	self.sceneObjects = {}		#scene objects
	
	#viz parameters
	self.vizObjects={}
        self.allinds = []
     	self.gridRadiusViz = 0
	
    @QtCore.pyqtSlot()
    def printModelViewMatrix(self):
        print self.modelview_matrix_
	

    def resizeGL(self, width, height):
        glViewport( 0, 0, width, height );
        self.set_projection( self.near_, self.far_, self.fovy_ );
        self.updateGL()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixd(self.modelview_matrix_)

    def set_projection(self, _near, _far, _fovy):
        self.near_ = _near
        self.far_ = _far
        self.fovy_ = _fovy
        self.makeCurrent()
        glMatrixMode( GL_PROJECTION )
        glLoadIdentity()
        gluPerspective( self.fovy_, float(self.width()) / max(1.0,float(self.height())),
                        self.near_, self.far_ )
        self.updateGL()

    def set_center(self, _cog):
        self.center_ = _cog
        self.view_all()

    def set_radius(self, _radius):
        self.radius_ = _radius
        self.set_projection(_radius / 100.0, _radius * 100.0, self.fovy_)
        self.reset_view()
        self.translate([0, 0, -_radius * 2.0])
        self.view_all()
        self.updateGL()

    def reset_view(self):
        # scene pos and size
        glMatrixMode( GL_MODELVIEW )
        glLoadIdentity()
        self.modelview_matrix_ = glGetDoublev( GL_MODELVIEW_MATRIX )
        self.set_center([0.0, 0.0, 0.0])

    def reset_rotation(self):
        self.modelview_matrix_[0] = [1.0, 0.0, 0.0, 0.0]
        self.modelview_matrix_[1] = [0.0, 1.0, 0.0, 0.0]
        self.modelview_matrix_[2] = [0.0, 0.0, 1.0, 0.0]
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixd(self.modelview_matrix_)
        self.updateGL()
   
    def translate(self, _trans):
        # Translate the object by _trans
        # Update modelview_matrix_
        self.makeCurrent()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslated(_trans[0], _trans[1], _trans[2])
        glMultMatrixd(self.modelview_matrix_)
        self.modelview_matrix_ = glGetDoublev(GL_MODELVIEW_MATRIX)
        self.translate_vector_[0] = self.modelview_matrix_[3][0]
        self.translate_vector_[1] = self.modelview_matrix_[3][1]
        self.translate_vector_[2] = self.modelview_matrix_[3][2]
        self.signalGLMatrixChanged.emit()

    def rotate(self, _axis, _angle):
    	#self.modelview_matrix_ = glGetDoublev(GL_MODELVIEW_MATRIX)
        t = [self.modelview_matrix_[0][0] * self.center_[0] +
             self.modelview_matrix_[1][0] * self.center_[1] +
             self.modelview_matrix_[2][0] * self.center_[2] +
             self.modelview_matrix_[3][0],
             self.modelview_matrix_[0][1] * self.center_[0] +
             self.modelview_matrix_[1][1] * self.center_[1] +
             self.modelview_matrix_[2][1] * self.center_[2] +
             self.modelview_matrix_[3][1],
             self.modelview_matrix_[0][2] * self.center_[0] +
             self.modelview_matrix_[1][2] * self.center_[1] +
             self.modelview_matrix_[2][2] * self.center_[2] +
             self.modelview_matrix_[3][2]]
	
        self.makeCurrent()
        glLoadIdentity()
        glTranslatef(t[0], t[1], t[2])
        glRotated(_angle, _axis[0], _axis[1], _axis[2])
        glTranslatef(-t[0], -t[1], -t[2])
        glMultMatrixd(self.modelview_matrix_)
        self.modelview_matrix_ = glGetDoublev(GL_MODELVIEW_MATRIX)
        self.signalGLMatrixChanged.emit()

    def view_all(self):
        self.translate( [ -( self.modelview_matrix_[0][0] * self.center_[0] +
                             self.modelview_matrix_[0][1] * self.center_[1] +
                             self.modelview_matrix_[0][2] * self.center_[2] +
                             self.modelview_matrix_[0][3]),
                           -( self.modelview_matrix_[1][0] * self.center_[0] +
                              self.modelview_matrix_[1][1] * self.center_[1] +
                              self.modelview_matrix_[1][2] * self.center_[2] +
                              self.modelview_matrix_[1][3]),
                           -( self.modelview_matrix_[2][0] * self.center_[0] +
                              self.modelview_matrix_[2][1] * self.center_[1] +
                              self.modelview_matrix_[2][2] * self.center_[2] +
                              self.modelview_matrix_[2][3] +
                              self.radius_ / 2.0 )])

    def map_to_sphere(self, _v2D):
        _v3D = [0.0, 0.0, 0.0]
        # inside Widget?
        if (( _v2D.x() >= 0 ) and ( _v2D.x() <= self.width() ) and
            ( _v2D.y() >= 0 ) and ( _v2D.y() <= self.height() ) ):
            # map Qt Coordinates to the centered unit square [-0.5..0.5]x[-0.5..0.5]
            x  = float( _v2D.x() - 0.5 * self.width())  / self.width()
            y  = float( 0.5 * self.height() - _v2D.y()) / self.height()

            _v3D[0] = x;
            _v3D[1] = y;
            # use Pythagoras to comp z-coord (the sphere has radius sqrt(2.0*0.5*0.5))
            z2 = 2.0*0.5*0.5-x*x-y*y;
            # numerical robust sqrt
            _v3D[2] = math.sqrt(max( z2, 0.0 ))

            # normalize direction to unit sphere
            n = linalg.norm(_v3D)
            _v3D = numpy.array(_v3D) / n

            return True, _v3D
        else:
            return False, _v3D

    def wheelEvent(self, _event):
        # Use the mouse wheel to zoom in/out
        d =  float(_event.delta()) / 200.0 * self.radius_
        if ((self.zpan+d) <= (self.near_+3) and (self.zpan+d) >= -1*(self.far_-10)):
            self.translate([0.0, 0.0, d])
            self.zpan = self.zpan + d
            self.updateGL()
            _event.accept()
	#print self.z_d

    def mousePressEvent(self, _event):
        self.last_point_2D_ = _event.pos()
        self.last_point_ok_, self.last_point_3D_ = self.map_to_sphere(self.last_point_2D_)
	if (_event.button()==1):
	    self.pressEventPicking()
	    
    def mouseMoveEvent(self, _event):
        newPoint2D = _event.pos()

        if ((newPoint2D.x() < 0) or (newPoint2D.x() > self.width()) or
            (newPoint2D.y() < 0) or (newPoint2D.y() > self.height())):
            return
        
        # Left button: rotate around center_
        # Middle button: translate object
        # Left & middle button: zoom in/out

        value_y = 0
        newPoint_hitSphere, newPoint3D = self.map_to_sphere(newPoint2D)

        dx = float(newPoint2D.x() - self.last_point_2D_.x())
        dy = float(newPoint2D.y() - self.last_point_2D_.y())

        w  = float(self.width())
        h  = float(self.height())

        # enable GL context
        self.makeCurrent()

        # move in z direction
        if (((_event.buttons() & QtCore.Qt.LeftButton) and (_event.buttons() & QtCore.Qt.MidButton))
            or (_event.buttons() & QtCore.Qt.LeftButton and _event.modifiers() & QtCore.Qt.ControlModifier)):
            value_y = self.radius_ * dy * 2.0 / h;
            self.translate([0.0, 0.0, value_y])
        # move in x,y direction
        elif (_event.buttons() & QtCore.Qt.MidButton
              or (_event.buttons() & QtCore.Qt.LeftButton and _event.modifiers() & QtCore.Qt.ShiftModifier)):
            z = - (self.modelview_matrix_[0][2] * self.center_[0] +
                   self.modelview_matrix_[1][2] * self.center_[1] +
                   self.modelview_matrix_[2][2] * self.center_[2] +
                   self.modelview_matrix_[3][2]) / (self.modelview_matrix_[0][3] * self.center_[0] +
                                                    self.modelview_matrix_[1][3] * self.center_[1] +
                                                    self.modelview_matrix_[2][3] * self.center_[2] +
                                                    self.modelview_matrix_[3][3])

	    
            fovy   = 45.0
            aspect = w / h
            n      = 0.01 * self.radius_
            up     = math.tan(fovy / 2.0 * math.pi / 180.0) * n
            right  = aspect * up

            self.xpan += 2.0 * dx / w * right / n * z
            self.ypan += -2.0 * dy / h * up / n * z

            self.translate( [2.0 * dx / w * right / n * z,
                             -2.0 * dy / h * up / n * z,
                             0.0] )

    
        # rotate
        elif (_event.buttons() & QtCore.Qt.LeftButton):
            if (not self.isInRotation_):
                self.isInRotation_ = True
                self.rotationBeginEvent.emit()
       
            axis = [0.0, 0.0, 0.0]
            angle = 0.0

            if (self.last_point_ok_ and newPoint_hitSphere):
                axis = numpy.cross(self.last_point_3D_, newPoint3D)
                cos_angle = numpy.dot(self.last_point_3D_, newPoint3D)
                if (abs(cos_angle) < 1.0):
                    angle = math.acos(cos_angle) * 180.0 / math.pi
                    angle *= 2.0
                self.rotate(axis, angle)
	
        # remember this point
        self.last_point_2D_ = newPoint2D
        self.last_point_3D_ = newPoint3D
        self.last_point_ok_ = newPoint_hitSphere

        # trigger redraw
        self.updateGL()

    
#additions by chaitanya from glwidget.py

    def enterEvent(self, ev):
	"""
	Event called when the mouse enters the widget area.
	"""
	self.grabKeyboard()

    def leaveEvent(self, ev):
	"""
	Event called when the mouse leaves the widget area.
	"""
	self.releaseKeyboard()

    def keyPressEvent(self, ev):
        """
        Key press callback.
        """
        key = str(ev.text()).upper()
		
        if (ev.modifiers() & QtCore.Qt.ControlModifier):
            self.ctrlPressed = True
#        elif (ev.modifiers() & QtCore.Qt.ShiftModifier):
#            self.shiftPressed = True
        elif (ev.key() == QtCore.Qt.Key_Up) or  (ev.key() == QtCore.Qt.Key_I):
            self.translate([0.0, 0.05, 0.0])
            self.ypan += 0.05
            self.updateGL()
        elif (ev.key() == QtCore.Qt.Key_Down) or  (ev.key() == QtCore.Qt.Key_K):
            self.translate([0.0, -0.05, 0.0])
            self.ypan += -0.05
            self.updateGL()
        elif (ev.key() == QtCore.Qt.Key_Left) or  (ev.key() == QtCore.Qt.Key_J):
            self.translate([-0.05, 0.0, 0.0])
            self.xpan += -0.05
            self.updateGL()
        elif (ev.key() == QtCore.Qt.Key_Right) or  (ev.key() == QtCore.Qt.Key_L):
            self.translate([0.05, 0.0, 0.0])
            self.xpan += 0.05
            self.updateGL()
        elif (ev.key() == QtCore.Qt.Key_Plus)or(ev.key() == QtCore.Qt.Key_PageUp)or(ev.key()==QtCore.Qt.Key_Period) or  (ev.key() == QtCore.Qt.Key_U):
            if ((self.zpan+0.75) <= self.near_+3):
                self.translate([0.0, 0.0, 0.75])
                self.zpan += 0.75
        	self.updateGL()
        elif (ev.key() == QtCore.Qt.Key_Minus)or(ev.key() == QtCore.Qt.Key_PageDown)or(ev.key()==QtCore.Qt.Key_Comma)or  (ev.key() == QtCore.Qt.Key_O):
            if ((self.zpan-0.75) >= -1*(self.far_-10)):
                self.translate([0.0, 0.0, -0.75])
                self.zpan += -0.75
                self.updateGL()
        elif (ev.key() == QtCore.Qt.Key_S):
            self.rotate([1.0, 0.0, 0.0],2.0)
            self.updateGL()
        elif (ev.key() == QtCore.Qt.Key_W):
            self.rotate([1.0, 0.0, 0.0],-2.0)
            self.updateGL()
        elif (ev.key() == QtCore.Qt.Key_D):
            self.rotate([0.0, 1.0, 0.0],2.0)
            self.updateGL()
        elif (ev.key() == QtCore.Qt.Key_A):
            self.rotate([0.0, 1.0, 0.0],-2.0)
            self.updateGL()
        elif (ev.key() == QtCore.Qt.Key_Q):
            self.rotate([0.0, 0.0, 1.0],2.0)
            self.updateGL()
        elif (ev.key() == QtCore.Qt.Key_E):
            self.rotate([0.0, 0.0, 1.0],-2.0)
            self.updateGL()

    def keyReleaseEvent(self, ev):
        """
        Key release callback.
        """
        if (ev.key() == QtCore.Qt.Key_Control):
            self.ctrlPressed = False
#        if (ev.key() == QtCore.Qt.Key_Shift):
#            self.shiftPressed = False

    def pressEventPicking(self):
	"""
	Picking event called when mousePressEvent() is called.
	Note that this event will be called before translationMoveEvent(),
	and so, it does not consider the translation event.
	"""
		
	pickedObject = self.tryPick()
	if pickedObject != None:
		# An object was picked.
	    if not(self.ctrlPressed):# and not(self.shiftPressed):
			# CTRL is not pressed.
    				
		if pickedObject in self.selectedObjects:
		# The picked object is already selected.
		# releaseEventPicking
		    pass
				
		else:
		# The picked object was not previously selected.
		# Deselect all previously selected objects.
	 	    self.selectedObjects.removeAll()

		# Select the picked object.
		    self.selectedObjects.add(pickedObject)
		    self.preSelectedObject = pickedObject


            # elif (self.shiftPressed):  #selects the parent instead, the whole cell for instance
            #     for objectSelected in self.sceneObjects:
            #         if (objectSelected.daddy == pickedObject.daddy):
            #             self.selectedObjects.add(objectSelected)

	    else:
		# CTRL is pressed.
			
		if pickedObject.selected:
		    pass
		else:
		    self.selectedObjects.add(pickedObject)
		    self.preSelectedObject = pickedObject
			
	else:
		# No objects were picked.
            self.selectedObjects.removeAll()
	#if selectedName: 	#prints the name of the selection if any
	#    print selectedName[0]
	#print self.selectedObjects	

	self.updateGL()

    def tryPick(self):
	"""
	Handles object picking, returning the object under the current mouse position.
	Returns None if there is none.
	"""
	if len(self.sceneObjects.keys()) == 0:
	    return None	#there are no objects on the glcanvas
	
	buffer = glSelectBuffer(len(self.sceneObjects.keys())*4)
	projection = glGetDouble(GL_PROJECTION_MATRIX)
	viewport = glGetInteger(GL_VIEWPORT)
		
	glRenderMode(GL_SELECT)
	glMatrixMode(GL_PROJECTION)
	glPushMatrix()
	glLoadIdentity()
	
	gluPickMatrix(self.last_point_2D_.x(),viewport[3]-self.last_point_2D_.y(),10,10,viewport)
	
	glMultMatrixf(projection)
	glInitNames()
	glPushName(0)
		
	glMatrixMode(GL_MODELVIEW)
	for i in range(len(self.sceneObjects.items())):
	    glLoadName(i)
	    self.sceneObjects.items()[i][1].render()
				
	glMatrixMode(GL_PROJECTION)
	glPopMatrix()
	glMatrixMode(GL_MODELVIEW)
	glFlush()
	glPopName()
			
	hits = glRenderMode(GL_RENDER)
	names = []		
	nearestHit = None
	for hit in hits:
	    near, far, names = hit
	    if (nearestHit == None) or (near < nearestHit[0]):
		nearestHit = [near, names[0]]
	if names:	
	    #print self.sceneObjectNames[names[0]]
	    self.compartmentSelected.emit(str(self.sceneObjects.items()[names[0]][0]))	#printing the cell path
        else:
            self.compartmentSelected.emit('') 

	if nearestHit != None:
	    return self.sceneObjects.items()[names[0]][1]

	return nearestHit


    def renderAxis(self):
	"""
	Creates the XYZ axis in the 0 coordinate, just for reference.
	"""
        glViewport(0,0,self.width()/8,self.height()/8)
        self.translate([-1*self.xpan,-1*self.ypan,-1*self.zpan])
        glMatrixMode(GL_MODELVIEW)

        # XYZ axis
	glLineWidth(2)
	glDisable(GL_LIGHTING)
	glBegin(GL_LINES)
	glColor(1, 0, 0)	#Xaxis, Red color
	glVertex3f(0, 0, 0)
	glVertex3f(0.15, 0, 0)
	glColor(0, 1, 0)	#Yaxis, Green color
	glVertex3f(0, 0, 0)
	glVertex3f(0, 0.15, 0)
	glColor(0, 0, 1)	#Zaxis, Blue color
	glVertex3f(0, 0, 0)
	glVertex3f(0, 0, 0.15)
	glEnd()
 
	glLineWidth(1)	

        glutInit()
        glColor(1,0,0)
        glRasterPos3f(0.16, 0.0, 0.0)
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13,88)#ascii x
        glColor(0,1,0)
        glRasterPos3f(0.0, 0.16, 0.0)
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13,89)#ascii y
        glColor(0,0,1)
        glRasterPos3f(0.0, 0.0, 0.16)
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13,90)#ascii z

        glEnable(GL_LIGHTING)

        self.translate([self.xpan,self.ypan,self.zpan])
        glViewport(0,0,self.width(),self.height())


#===============================================================================
#
# Local Variables:
# mode: Python
# indent-tabs-mode: nil
# End:
#
#==============================================================================
