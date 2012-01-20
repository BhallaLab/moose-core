#Author:Chaitanya CH
#FileName: objects.py

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

from OpenGL.GL import *
from OpenGL.raw.GLUT import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from numpy import sqrt,arccos

class BaseObject(object):
	"""
	Base class for any object used in the GLWidget class.
	"""
	
	def __init__(self, parent):
		"""
		Constructor of the base class. Usually, it should be called in
		the constructor of the inherited classes.
		"""
		#default placement of the drawing.
		self._centralPos = [0.0,0.0,0.0]
		self.rotation = [0.0,0.0,0.0,0.0]
		# Indicates whether the object is selected or not.
		self.selected = False
		
		# Initial color of the object.
		self.r, self.g, self.b = 1.0, 0.0, 0.0
		self.oldColor = self.r, self.g, self.b  
		# Reference to the GLWidget object that contains this object.
		self.parent = parent
		self.daddy = ''	#can have only one daddy obviously.
		self.name = ''
		self.kids = []
		#subdivisions of cylinder,disk,sphere x by x
		self.subdivisions = 10

#	@property
	def getCentralPosition(self):
		return self._centralPos
		
#	@centralPosition.setter
	def setCentralPosition(self, value):
		self._centralPos = value

	def getRotation(self):
		return self.rotation

	def setRotation(self, value):
		self.rotation = value
			
	def getName(self):
		return self.name

	def setProperties(self,centralPos,rotation,r,g,b):
		self._centralPos = centralPos	
		self.rotation = rotation	
		self.r = r
		self.g = g
		self.b = b

	def render(self):
		"""
		Virtual method that should be overridden in base-classes.
		Method called when the object should be drawn.
		"""
		pass
			
	def select(self, newStatus):
		"""
		Selects or unselects the object, depending on the newStatus argument.
		Also changes the object's color according to the selection status.
		"""
		
		self.selected = newStatus
		if self.selected:
			self.oldColor = [self.r,self.g,self.b]
			self.r, self.g, self.b = 0, 1, 0
		else:
			self.r, self.g, self.b = self.oldColor
		

class somaSphere(BaseObject):
	"""
	Class that defines a compartment as a sphere. drawn only when x=x0&y=y0&z=z0
	"""
	
	def __init__(self,parent,name,cellName,l_coords,cellCentre=[0.0,0.0,0.0],cellAngle=[0.0,0.0,0.0,0.0]):
		"""
		Constructor.
		"""
		super(somaSphere, self).__init__(parent)
		self.name = name
		self.radius = l_coords[6]/2
		self.centre = [l_coords[0],l_coords[1],l_coords[2]]
		self.daddy  = cellName
		self._centralPos = cellCentre
		self.rotation = cellAngle
			
	def render(self):
		"""
		Renders the sphere.
		"""
		glutInit(1,1)
		glPushMatrix()
		glColor(self.r, self.g, self.b)
		
		glTranslate(*self._centralPos[:3])		#move pen to given location
		glTranslate(*self.centre[:3])			#mid point of the compartment line

		gluSphere(gluNewQuadric(),self.radius, 10, 10)

		glTranslate(*[i*-1 for i in self.centre[:3]])	#bring back pen to origin and orientation
		glTranslate(*[i*-1 for i in self._centralPos[:3]])

		glPopMatrix()
		
		
class somaDisk(BaseObject):
	"""
	Class that defines a compartment as a disk. only when x=x0&y=y0&z=z0
	"""
	
	def __init__(self,parent,name,cellName,l_coords,cellCentre=[0.0,0.0,0.0],cellAngle=[0.0,0.0,0.0,0.0]):
		"""
		Constructor.
		"""
		super(somaDisk, self).__init__(parent)
		self.name = name
		self.radius = l_coords[6]/2
		self.centre = [l_coords[0],l_coords[1],l_coords[2]]
		self.daddy  = cellName
		self._centralPos = cellCentre
		self.rotation = cellAngle

			
	def render(self):
		"""
		Renders the soma as a disk.
		"""
		glutInit(1,1)
		glPushMatrix()
		glColor(self.r, self.g, self.b)
		
		glRotate(*self.rotation[:4])				#move pen to the given orientation, absolute coordinate =[0,0,0,0]
		glTranslate(*self._centralPos[:3])			#move pen to the given location, absolute coordinate =[0,0,0]
		glTranslate(*self.centre[:3])				#mid point of the compartment line
		
		quadric = gluNewQuadric()
		gluDisk( quadric, 0.0, self.radius, 10, 1)
		
		glTranslate(*[i*-1 for i in self.centre[:3]])
		glTranslate(*[i*-1 for i in self._centralPos[:3]])	#bring back to origin
		glRotate(*[i*-1 for i in self.rotation[:4]])		#bring back to original orientation
		glPopMatrix()

class cLine(BaseObject):
	"""
	Class that defines a compartment as a simple line.
	"""
	
	def __init__(self,parent,name,cellName,l_coords,cellCentre=[0.0,0.0,0.0],cellAngle=[0.0,0.0,0.0,0.0]):
		"""
		Constructor.
		"""
		super(cLine, self).__init__(parent)
		self.name = name
		self.l_coords = l_coords
		self.daddy = cellName
		self._centralPos = cellCentre
		self.rotation = cellAngle

	def render(self):
		"""
		Renders the compartment line.
		"""
		glPushMatrix()
		glColor(self.r, self.g, self.b)
		glRotate(*self.rotation[:4])		#get pen to the new orientation
		glTranslate(*self._centralPos[:3])	#get pen to the new position
		
		glLineWidth(2)
		glDisable(GL_LIGHTING)
		glBegin(GL_LINES)
  	    	glVertex3f(self.l_coords[0],self.l_coords[1],self.l_coords[2])
	    	glVertex3f(self.l_coords[3],self.l_coords[4],self.l_coords[5])
	    	glEnd()		
	    	glEnable(GL_LIGHTING)
		
		glTranslate(*[i*-1 for i in self._centralPos[:3]])	#get pen back to the origin
		glRotate(*[i*-1 for i in self.rotation[:4]])		#get pen to the original orientation
		
		glPopMatrix()	


class cCylinder(BaseObject):
	"""
	Class that defines a compartment as a cylindrical shape. 
	"""
	
	def __init__(self,parent,name,cellName,l_coords,cellCentre=[0.0,0.0,0.0],cellAngle=[0.0,0.0,0.0,0.0]):
		"""
		Constructor.
		"""
		super(cCylinder, self).__init__(parent)
		self.name = name
		self.l_coords = l_coords
		self.daddy = cellName
		self._centralPos = cellCentre
		self.rotation = cellAngle


	def render(self):
		"""
		Renders the compartment as a cylinder.
		"""
		x1,y1,z1,x2,y2,z2 = self.l_coords[:6]
		radius = self.l_coords[6]/2
		
		vx = x2-x1
  		vy = y2-y1
  		vz = z2-z1

  		if(vz == 0.0):
			vz = .001 #causes trouble sometimes

		v = sqrt( vx*vx + vy*vy + vz*vz )
 		ax = 57.2957795*arccos( vz/v )
  		if ( vz < 0.0 ):
      			ax = -ax
  		rx = -vy*vz
		ry = vx*vz
  		glPushMatrix()
		glColor(self.r, self.g, self.b)
		
		glRotate(*self.rotation[:4]) 		#get pen to set orientation, in absolute coordinates [0,0,0,0].
		glTranslate(*self._centralPos[:3])	#if absolute coordinates [0,0,0]
		
  		glTranslatef(x1,y1,z1 )
  		glRotatef(ax,rx,ry,0.0)
  		
  		quadric = gluNewQuadric()
  		gluQuadricNormals(quadric, GLU_SMOOTH)
  		
  		gluQuadricOrientation(quadric,GLU_OUTSIDE)
  		gluCylinder(quadric, radius, radius, v, self.subdivisions, 1)
  		
  		gluQuadricOrientation(quadric,GLU_INSIDE)
  		gluDisk( quadric, 0.0, radius, self.subdivisions, 1)
  		glTranslatef( 0,0,v )
  		
  		gluQuadricOrientation(quadric,GLU_OUTSIDE)
  		gluDisk( quadric, 0.0, radius, self.subdivisions, 1)
  		
  		glTranslate(*[i*-1 for i in self._centralPos[:3]])	#bring pen back to origin.
		glRotate(*[i*-1 for i in self.rotation[:4]])		#bring back to original orientation
  		
  		glPopMatrix()	
 

class cCapsule(BaseObject):
	"""
	Class that defines a compartment as a capsule shape. A cylinder with the base replaced by a sphere.
	"""
	
	def __init__(self,parent,name,cellName,l_coords,cellCentre=[0.0,0.0,0.0],cellAngle=[0.0,0.0,0.0,0.0]):
		"""
		Constructor.
		"""
		super(cCapsule, self).__init__(parent)
		self.name = name
		self.l_coords = l_coords
		self.daddy = cellName
		self._centralPos = cellCentre
		self.rotation = cellAngle


	def render(self):
		"""
		Renders the compartment as a cylinder.
		"""
		x1,y1,z1,x2,y2,z2 = self.l_coords[:6]
		radius = self.l_coords[6]/2
		
		vx = x2-x1
  		vy = y2-y1
  		vz = z2-z1

  		if(vz == 0.0):
			vz = .001 #causes trouble sometimes

		v = sqrt( vx*vx + vy*vy + vz*vz )
 		ax = 57.2957795*arccos( vz/v )
  		if ( vz < 0.0 ):
      			ax = -ax
  		rx = -vy*vz
		ry = vx*vz
  		glPushMatrix()
		glColor(self.r, self.g, self.b)
		
		glRotate(*self.rotation[:4]) 		#get pen to set orientation, in absolute coordinates [0,0,0,0].
		glTranslate(*self._centralPos[:3])	#if absolute coordinates [0,0,0]
		
  		glTranslatef(x1,y1,z1 )
  		glRotatef(ax,rx,ry,0.0)
  		
  		quadric = gluNewQuadric()
  		gluQuadricNormals(quadric, GLU_SMOOTH)
  		
  		gluQuadricOrientation(quadric,GLU_OUTSIDE)
  		gluCylinder(quadric, radius, radius, v, self.subdivisions, 1)
  		
  		gluQuadricOrientation(quadric,GLU_INSIDE)
  		gluSphere(gluNewQuadric(),radius, self.subdivisions, self.subdivisions)
  		glTranslatef( 0,0,v )
  		
  		gluQuadricOrientation(quadric,GLU_OUTSIDE)
  		gluSphere(gluNewQuadric(),radius, self.subdivisions, self.subdivisions)
  		
  		glTranslate(*[i*-1 for i in self._centralPos[:3]])	#bring pen back to origin.
		glRotate(*[i*-1 for i in self.rotation[:4]])		#bring back to original orientation
  		
  		glPopMatrix() 
 
			
