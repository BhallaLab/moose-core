#Author:Chaitanya CH
#FileName: group.py

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
		
