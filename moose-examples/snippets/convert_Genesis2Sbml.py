# convert_Genesis2Sbml.py --- 
# 
# Filename: convert_Genesis2Sbml.py
# Description: 
# Author:Harsha Rani  
# Maintainer: 
# Created: Mon Jan 19 09:16:58 2015 (+0530)
# Version: 
# Last-Updated: Thr Dec 24 15:155:38 2012 (+0530)
#           By: Harsha Rani
#     Update #: 
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# The script demonstates to convert Chemical (Genesis) file to SBML file using moose
# 
# 

# Change log:
# 
# 
# 
# 
# This program is free software; you can redistribute it and/or
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
# 
# 

# Code:

import moose
def main():
	"""This example illustrates loading a kinetic model defined in Genesis format
	into Moose using loadModel function and using writeSBML function
	one can save the model into SBML format. \n
	Moose needs to be compiled with libsbml
"""
	#This command loads the file into the path '/Kholodenko'
	moose.loadModel('../genesis/Kholodenko.g','/Kholodenko')
	
	#Writes model to xml file
	moose.writeSBML('/Kholodenko','Kholodenko_tosbml.xml')

if __name__ == '__main__':
	main()
