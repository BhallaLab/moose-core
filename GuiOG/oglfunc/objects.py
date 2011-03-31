from OpenGL.GL import *
from OpenGL.raw.GLUT import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from numpy import sqrt,arccos
from moose import Compartment as mcc	#because of detecting soma.

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
		
		# Reference to the GLWidget object that contains this object.
		self.parent = parent
		self.daddy = []
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
			self.r, self.g, self.b = 0, 1, 0
		else:
			self.r, self.g, self.b = 1, 0, 0
		
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
		glRotate(*self.rotation[:4])
		glTranslate(*self._centralPos[:3])
		#glMultMatrixf(self.rotation)
		glLineWidth(2)
		glBegin(GL_LINES)
  	    	glVertex3f(self.l_coords[0],self.l_coords[1],self.l_coords[2])
	    	glVertex3f(self.l_coords[3],self.l_coords[4],self.l_coords[5])
	    	glEnd()		
		#glTranslate(*[i*-1 for i in self._centralPos[:3]])
		#glRotate(*[i*-1 for i in self.rotation[:4]])
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
			if (mcc(l_coords[i][7]).name=='soma'):	#moose.Compartment =mcc
				compartmentLine = somaSphere(self,l_coords[i],cellName)
			else:
	    			if style==1:
	    				compartmentLine = cLine(self,l_coords[i],cellName)
				else: #style==2
					compartmentLine = cCylinder(self,l_coords[i],cellName)
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
			glPushMatrix()
			glColor(self.r, self.g, self.b)
			glRotate(*self.rotation[:4])
			glTranslate(*self._centralPos[:3])
			#glMultMatrixf(self.rotation)
			
			for obj in self.kids:
				obj.setCellParentProps(self._centralPos,self.rotation,self.r, self.g, self.b)
				obj.render()
			#glTranslate(*[i*-1 for i in self._centralPos[:3]])
			#glRotate(*[i*-1 for i in self.rotation[:4]])
			glPopMatrix()
			

class somaSphere(BaseObject):
	"""
	Class that defines a sphere.
	"""
	
	def __init__(self, parent,l_coords,cellName=[]):
		"""
		Constructor.
		"""
		super(somaSphere, self).__init__(parent)
		self.radius = (sqrt((l_coords[0]-l_coords[3])**2+(l_coords[1]-l_coords[4])**2+(l_coords[2]-l_coords[5])**2))/2
		self.centre = [(l_coords[0]+l_coords[3])/2,(l_coords[1]+l_coords[4])/2,(l_coords[2]+l_coords[5])/2]
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
		glRotate(*self.rotation[:4])
		glTranslate(*self._centralPos[:3])
		#glMultMatrixf(self.rotation)
		glTranslate(*self.centre[:3])
		gluSphere(gluNewQuadric(),self.radius, 20, 20)
		#glTranslate(*[i*-1 for i in self.centre[:3]])
		#glTranslate(*[i*-1 for i in self._centralPos[:3]])
		#glRotate(*[i*-1 for i in self.rotation[:4]])
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
		subdivisions = 20
		
		vx = x2-x1
  		vy = y2-y1
  		vz = z2-z1

  		if(vz == 0):
      			vz = .0001

  		v = sqrt( vx*vx + vy*vy + vz*vz )
 		ax = 57.2957795*arccos( vz/v )
  		if ( vz < 0.0 ):
      			ax = -ax
  		rx = -vy*vz
		ry = vx*vz
  		glPushMatrix()
		glColor(self.r, self.g, self.b)
  		glTranslatef( x1,y1,z1 )
  		glRotatef(ax, rx, ry, 0.0)
  		
  		quadric = gluNewQuadric()
  		gluQuadricNormals(quadric, GLU_SMOOTH)
  		
  		gluQuadricOrientation(quadric,GLU_OUTSIDE)
  		gluCylinder(quadric, radius, radius, v, subdivisions, 1)
  		
  		gluQuadricOrientation(quadric,GLU_INSIDE)
  		gluDisk( quadric, 0.0, radius, subdivisions, 1)
  		
  		glTranslatef( 0,0,v )
  		
  		gluQuadricOrientation(quadric,GLU_OUTSIDE)
  		gluDisk( quadric, 0.0, radius, subdivisions, 1)
  		glPopMatrix()	
