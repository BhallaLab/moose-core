from OpenGL.GL import *
from OpenGL.raw.GLUT import *

class BaseObject(object):
	"""
	Base class for any object used in the GLWidget class.
	"""
	
	def __init__(self, parent):
		"""
		Constructor of the base class. Usually, it should be called in
		the constructor of the inherited classes.
		"""
		
		# Indicates whether the object is selected or not.
		self.selected = False
		
		# Initial color of the object.
		self.r, self.g, self.b = 1.0, 0.0, 0.0
		
		# Reference to the GLWidget object that contains this object.
		self.parent = parent
			
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
	Class that defines a sphere.
	"""
	
	def __init__(self, parent,lcoords):
		"""
		Constructor.
		"""
		super(cLine, self).__init__(parent)
		self.l_coords = lcoords
		
	def render(self):
		"""
		Renders the sphere.
		"""
		
		glPushMatrix()
		glColor(self.r, self.g, self.b)
		glLineWidth(2)
		glBegin(GL_LINES)
		
  	    	glVertex3f(self.l_coords[0],self.l_coords[1],self.l_coords[2])
	    	glVertex3f(self.l_coords[3],self.l_coords[4],self.l_coords[5])
	    	glEnd()		

		glPopMatrix()	
