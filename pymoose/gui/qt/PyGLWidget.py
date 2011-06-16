#Author:Chaitanya CH
#FileName: PyGLWidget.py (modified from Arne Schmitz code)

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
from oglfunc.group import *

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
       
        #self.toolbar=QtGui.QToolBar(self)
        #self.toolbar.setGeometry(10,20,200,40)
        #self.toolbar.setFloatable(False)
        #self.toolbar.setMovable(True)
        #self.toolbar.addSeparator()
        #self.toolbar.show()
	#self.toolbar.raise_()
        #self.catbutt = QtGui.QToolButton(self.toolbar)
        #self.catbutt.setGeometry(10,0,40,40)
        #self.catbutt.setIcon(QtGui.QIcon("Search.png"))
        #self.catbutt2 = QtGui.QToolButton(self.toolbar)
        #self.catbutt2.setGeometry(50,0,40,40)
        #self.catbutt2.setIcon(QtGui.QIcon("resize.png"))
        #self.toolbar.allowedAreas(Qt.BottomToolBarArea)
        
        
        self.setMouseTracking(True)

        self.modelview_matrix_  = []
        self.translate_vector_  = [0.0, 0.0, 0.0]
        self.viewport_matrix_   = []
        self.projection_matrix_ = []
        self.near_   = 1.0		#0.1
        self.far_    = 500.0
        self.fovy_   = 4.0
        self.radius_ = 10.0
        self.last_point_2D_ = QtCore.QPoint()
        self.last_point_ok_ = False
        self.last_point_3D_ = [1.0, 0.0, 0.0]
        self.isInRotation_  = False
	
	#additions by chaitanya
	
	self.lights = 1			#lights	
	self.ctrlPressed = False 	#default no control pressed
	self.selectedObjects =Group(self)		#each line is a scene object.
	self.sceneObjects = []		#scene objects, abstraction depends on the selection mode.
	self.sceneObjectNames = []	#names of the scene objects being drawn
	self.selectionMode = 0 		#select compartments by default =1 selects compartments.
	
	#viz parameters
	self.viz=0
	self.vizObjectNames=[]
	self.vizObjects=[]

    	self.specificCompartmentName = 'soma'
    	self.gridRadiusViz = 0
	
    @QtCore.pyqtSlot()
    def printModelViewMatrix(self):
        print self.modelview_matrix_
	

    def initializeGL(self):
        # OpenGL state
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glEnable(GL_DEPTH_TEST)
	self.reset_view()

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
        #if self.height ==0:
        #	self.height=1
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
        d = - float(_event.delta()) / 200.0 * self.radius_
	self.translate([0.0, 0.0, d])
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
	elif (ev.key() == QtCore.Qt.Key_Up):
		self.translate([0.0, 0.25, 0.0])
		self.updateGL()
	elif (ev.key() == QtCore.Qt.Key_Down):
		self.translate([0.0, -0.25, 0.0])
		self.updateGL()
	elif (ev.key() == QtCore.Qt.Key_Left):
		self.translate([-0.25, 0.0, 0.0])
		self.updateGL()
	elif (ev.key() == QtCore.Qt.Key_Right):
		self.translate([0.25, 0.0, 0.0])
		self.updateGL()
	elif (ev.key() == QtCore.Qt.Key_Plus)or(ev.key() == QtCore.Qt.Key_PageUp)or(ev.key()==QtCore.Qt.Key_Period):
		self.translate([0.0, 0.0, 0.75])
		self.updateGL()
	elif (ev.key() == QtCore.Qt.Key_Minus)or(ev.key() == QtCore.Qt.Key_PageDown)or(ev.key()==QtCore.Qt.Key_Comma):
		self.translate([0.0, 0.0, -0.75])
		self.updateGL()
	elif (ev.key() == QtCore.Qt.Key_A):
		self.rotate([1.0, 0.0, 0.0],2.0)
		self.updateGL()
	elif (ev.key() == QtCore.Qt.Key_Q):
		self.rotate([1.0, 0.0, 0.0],-2.0)
		self.updateGL()
	elif (ev.key() == QtCore.Qt.Key_U):
		self.rotate([0.0, 1.0, 0.0],2.0)
		self.updateGL()
	elif (ev.key() == QtCore.Qt.Key_Y):
		self.rotate([0.0, 1.0, 0.0],-2.0)
		self.updateGL()
	elif (ev.key() == QtCore.Qt.Key_Z):
		self.rotate([0.0, 0.0, 1.0],2.0)
		self.updateGL()
	elif (ev.key() == QtCore.Qt.Key_X):
		self.rotate([0.0, 0.0, 1.0],-2.0)
		self.updateGL()

    def keyReleaseEvent(self, ev):
	"""
	Key release callback.
	"""
	if (ev.key() == QtCore.Qt.Key_Control):
	    self.ctrlPressed = False


    def pressEventPicking(self):
	"""
	Picking event called when mousePressEvent() is called.
	Note that this event will be called before translationMoveEvent(),
	and so, it does not consider the translation event.
	"""
		
	pickedObject = self.tryPick()
	if pickedObject != None:
		# An object was picked.
	    if not(self.ctrlPressed):
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
	if len(self.sceneObjects) == 0:
	    return None	#there are no objects on the glcanvas
	
	buffer = glSelectBuffer(len(self.sceneObjects)*4)
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
	for i in range(len(self.sceneObjects)):
	    glLoadName(i)
	    self.sceneObjects[i].render()
				
	glMatrixMode(GL_PROJECTION)
	glPopMatrix()
	glMatrixMode(GL_MODELVIEW)
	glFlush()
	glPopName()
			
	hits = glRenderMode(GL_RENDER)
	names =[]		
	nearestHit = None
	for hit in hits:
	    near, far, names = hit
	    if (nearestHit == None) or (near < nearestHit[0]):
		nearestHit = [near, names[0]]
	if names:	
	    #print self.sceneObjectNames[names[0]]
	    self.compartmentSelected.emit(str(self.sceneObjectNames[names[0]]))	#printing the cell path
	    #self.compartmentSelected.emit(QtCore.SIGNAL('PyQt_PyObject'),self.sceneObjectNames[names[0]])
	if nearestHit != None:
	    return self.sceneObjects[nearestHit[1]]

	return nearestHit


    def renderAxis(self):
	"""
	Creates the XYZ axis in the 0 coordinate, just for reference.
	"""
	# XYZ axis
	glLineWidth(1)
	glDisable(GL_LIGHTING)
	glBegin(GL_LINES)
	glColor(1, 0, 0)	#Xaxis, Red color
	glVertex3f(0, 0, 0)
	glVertex3f(1, 0, 0)
	glColor(0, 1, 0)	#Yaxis, Green color
	glVertex3f(0, 0, 0)
	glVertex3f(0, 1, 0)
	glColor(0, 0, 1)	#Zaxis, Blue color
	glVertex3f(0, 0, 0)
	glVertex3f(0, 0, 1)
	glEnd()
	glEnable(GL_LIGHTING)
	glLineWidth(1)	

    

#===============================================================================
#
# Local Variables:
# mode: Python
# indent-tabs-mode: nil
# End:
#
#===============================================================================
