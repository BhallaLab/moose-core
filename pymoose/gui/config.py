# config.py --- 
# 
# Filename: config.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Sat Feb 13 16:07:56 2010 (+0530)
# Version: 
# Last-Updated: Sat Feb 13 17:19:11 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 42
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
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

GL_CLIENT_EXECUTABLE = '../../gl/src/glclient'
GL_COLORMAP_DIR = '../../gl/colormaps'
GL_COLORMAP_RAINBOW2 = 'rainbow2'
GL_COLORMAP_HOT = 'hot'
GL_COLORMAP_GREY = 'grey'
GL_COLORMAP_REDHOT = 'redhot'

GL_PORT = '9999'

from PyQt4.Qt import Qt
from PyQt4 import QtGui, QtCore

settings = None
def getSettings():
    '''Initializes the QSettings for the application and returns it.'''
    global settings
    if not settings:
	QtCore.QCoreApplication.setOrganizationName('NCBS')
	QtCore.QCoreApplication.setOrganizationDomain('ncbs.res.in')
	QtCore.QCoreApplication.setApplicationName('MOOSE')
    settings = QtCore.QSettings()
    return settings


# 
# config.py ends here
