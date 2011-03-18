import numpy

class Group(object):
	"""
	This class represents a group of objects.
	"""

	def __init__(self, parent):
		"""
		Creates a new Group object.
		"""
		
		# List of objects in this group.
		self._objects = []
		
		
	def __iter__(self):
		"""
		Object that will have the iteration items.
		"""
		
		return self._objects.__iter__()
	
	def __len__(self):
		"""
		Returns how many objects there are in this group.
		"""
		
		return len(self._objects)
	
	
	def add(self, object, autoSelect=True):
		"""
		Adds an object to the group.
		"""
		
		newPts = None
				
		self._objects.append(object)
		
		if autoSelect:
			object.select(True)
		
		
	def remove(self, object, autoDeselect=True):
		"""
		Removes an object from the group.
		"""
		
		self._objects.remove(object)
		
		if autoDeselect:
			object.select(False)
		
		
	def removeAll(self):
		"""
		Removes all objects from the group.
		"""
		
		for obj in self._objects:
			obj.select(False)
			
		del self._objects[:]

	def render(self, pickingMode=False):
		"""
		Renders the group effects. Does not render the objects.
		"""
		
		if len(self._objects) == 0:
			return
		
