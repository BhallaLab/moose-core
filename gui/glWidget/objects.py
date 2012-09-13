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
from numpy import sqrt,arccos,arctan,absolute

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
		self.kids = []

	@property
	def centralPosition(self):
		return self._centralPos
		
	@centralPosition.setter
	def centralPosition(self, value):
		self._centralPos = value

			
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
		
class cLine(BaseObject):
	"""
	Class that defines a compartment line.
	"""
	
	def __init__(self, parent,l_coords,cellName=[]):
		"""
		Constructor.
		"""
		super(cLine, self).__init__(parent)
		self.l_coords = l_coords
		self.daddy = cellName


	def setCellParentProps(self,centralPos,rotation,r,g,b):
		self._centralPos = centralPos	
		self.rotation = rotation	
		self.r = r
		self.g = g
		self.b = b	

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

class cellStruct(BaseObject):
	"""
	Class that defines a cellstructure.
	"""
	def __init__(self, parent,l_coords,cellName,style,cropName=[]):
		"""
		Constructor.
		"""
		super(cellStruct, self).__init__(parent)
		self.daddy = cropName
		
		for i in range(0,len(l_coords),1):	
			if (l_coords[i][0] == l_coords[i][3] and l_coords[i][1] == l_coords[i][4] and l_coords[i][2] == l_coords[i][5]):
				if style==0:	#simple disk model
					compartmentLine = somaDisk(self,l_coords[i],cellName)
					self.kids.append(compartmentLine)
				elif (style==1)or(style==2):
					compartmentLine = somaSphere(self,l_coords[i],cellName)
				elif style==3:
					compartmentLine = somaDisk(self,[0,0,0,0,0,0,l_coords[i][7]],cellName)
					compartmentLine.radius = 0.20
					self.kids.append(compartmentLine)
			else:
	    			if style==1:	#ball and stick model
	    				compartmentLine = cLine(self,l_coords[i],cellName)
				elif style==2:	#realistic model
					compartmentLine = cCylinder(self,l_coords[i],cellName)
				
			if (style==1)or(style==2):
				self.kids.append(compartmentLine)	

    	
	def setCropParentProps(self,centralPos,rotation,r,g,b):
		self._centralPos = centralPos	
		self.rotation	
		self.r = r
		self.g = g
		self.b = b	


	def render(self):
		"""
		Renders the cell structure.
		"""

		if self.kids:
			glRotate(*self.rotation[:4])			#move pen to given orientation
			glTranslate(*self._centralPos[:3])		#move pen to given location
			for obj in self.kids:
				obj.setCellParentProps([0.0,0.0,0.0],[0.0,0.0,0.0,0.0],self.r, self.g, self.b)	#need not have to move the component objects again.
				obj.render()
			glTranslate(*[i*-1 for i in self._centralPos[:3]]) 	#bring back pen to origin and orientation
			glRotate(*[i*-1 for i in self.rotation[:4]])

			

class somaSphere(BaseObject):
	"""
	Class that defines a sphere.
	"""
	
	def __init__(self, parent,l_coords,cellName=[]):
		"""
		Constructor.
		"""
		super(somaSphere, self).__init__(parent)
		#self.radius = (sqrt((l_coords[0]-l_coords[3])**2+(l_coords[1]-l_coords[4])**2+(l_coords[2]-l_coords[5])**2))/2
		self.radius = l_coords[6]/2
		self.centre = [l_coords[0],l_coords[1],l_coords[2]]
		self.daddy  = cellName

	def setCellParentProps(self,centralPos,rotation,r,g,b):
		self._centralPos = centralPos	
		self.rotation = rotation	
		self.r = r
		self.g = g
		self.b = b
		
			
	def render(self):
		"""
		Renders the sphere.
		"""
		glutInit(1,1)
		glPushMatrix()
		glColor(self.r, self.g, self.b)
		
		glRotate(*self.rotation[:4])			#move pen to given orientation
		glTranslate(*self._centralPos[:3])		#move pen to given location
		glTranslate(*self.centre[:3])			#mid point of the compartment line

		gluSphere(gluNewQuadric(),self.radius, 9, 9)

		glTranslate(*[i*-1 for i in self.centre[:3]])	#bring back pen to origin and orientation
		glTranslate(*[i*-1 for i in self._centralPos[:3]])
		glRotate(*[i*-1 for i in self.rotation[:4]])

		glPopMatrix()
		
		
class somaDisk(BaseObject):
	"""
	Class that defines a sphere.
	"""
	
	def __init__(self, parent,l_coords,cellName=[]):
		"""
		Constructor.
		"""
		super(somaDisk, self).__init__(parent)
		#self.radius = (sqrt((l_coords[0]-l_coords[3])**2+(l_coords[1]-l_coords[4])**2+(l_coords[2]-l_coords[5])**2))/2
		self.radius = l_coords[6]/2
		self.centre = [l_coords[0],l_coords[1],l_coords[2]]
		self.daddy  = cellName

	def setCellParentProps(self,centralPos,rotation,r,g,b):
		self._centralPos = centralPos	
		self.rotation = rotation	
		self.r = r
		self.g = g
		self.b = b
		
			
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
		gluDisk( quadric, 0.0, self.radius, 30, 1)
		
		glTranslate(*[i*-1 for i in self.centre[:3]])
		glTranslate(*[i*-1 for i in self._centralPos[:3]])	#bring back to origin
		glRotate(*[i*-1 for i in self.rotation[:4]])		#bring back to original orientation
		glPopMatrix()


class cCylinder(BaseObject):
	"""
	Class that defines a compartment line.
	"""
	
	def __init__(self, parent,l_coords,cellName=[]):
		"""
		Constructor.
		"""
		super(cCylinder, self).__init__(parent)
		self.l_coords = l_coords
		self.daddy = cellName

	def setCellParentProps(self,centralPos,rotation,r,g,b):
		self._centralPos = centralPos	
		self.rotation = rotation	
		self.r = r
		self.g = g
		self.b = b	

	def render(self):
		"""
		Renders the compartment as a cylinder.
		"""
		x1,y1,z1,x2,y2,z2 = self.l_coords[:6]
		radius = self.l_coords[6]/2
		subdivisions = 5
		quadric = gluNewQuadric()
		vx = x2-x1
		vy = y2-y1
		vz = z2-z1

		#float ax,rx,ry,rz;
		length = sqrt( vx*vx + vy*vy + vz*vz )

		glPushMatrix()

		glColor(self.r, self.g, self.b)
		glRotate(*self.rotation[:4]) 		#get pen to set orientation, in absolute coordinates [0,0,0,0].
		glTranslate(*self._centralPos[:3])	#if absolute coordinates [0,0,0]

		glTranslatef( x1,y1,z1 )
		if (absolute(vz) < 0.0001):
			glRotatef(90, 0,1,0)
			if vx == 0:
				if vy < 0:
					ax = 57.2957795
				else:
					ax = -57.2957795
			else:
				ax = 57.2957795*-arctan( vy / vx )
			if (vx < 0):
				ax = ax + 180
			rx = 1
			ry = 0
			rz = 0
		else:
			ax = 57.2957795*arccos( vz/ length )
			if (vz < 0.0):
				ax = -ax
			rx = -vy*vz
			ry = vx*vz
			rz = 0

		v = sqrt( vx*vx + vy*vy + vz*vz )

		glRotatef(ax, rx, ry, rz)
		gluQuadricOrientation(quadric,GLU_OUTSIDE)
		gluCylinder(quadric, radius, radius, length, subdivisions, 1)

		gluQuadricOrientation(quadric,GLU_INSIDE)
		gluDisk( quadric, 0.0, radius, subdivisions, 1)

		glTranslatef( 0,0,v )
		
		gluQuadricOrientation(quadric,GLU_OUTSIDE)
		gluDisk( quadric, 0.0, radius, subdivisions, 1)

  		glTranslate(*[i*-1 for i in self._centralPos[:3]])	#bring pen back to origin.
		glRotate(*[i*-1 for i in self.rotation[:4]])		#bring back to original orientation
		glPopMatrix()


