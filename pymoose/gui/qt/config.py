# config.py --- 
# 
# Filename: config.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Sat Feb 13 16:07:56 2010 (+0530)
# Version: 
# Last-Updated: Thu Jul 22 15:38:34 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 116
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

GL_CLIENT_EXECUTABLE = 'glclient'
GL_COLORMAP_DIR = 'colormaps'
GL_COLORMAP_RAINBOW2 = 'rainbow2'
GL_COLORMAP_HOT = 'hot'
GL_COLORMAP_GREY = 'grey'
GL_COLORMAP_REDHOT = 'redhot'
GL_DEFAULT_COLORMAP = 'colormaps/rainbow2'
GL_PORT = '9999'

import logging
from PyQt4.Qt import Qt
from PyQt4 import QtGui, QtCore

settings = None

KEY_FIRSTTIME = 'firsttime'
# KEY_STATE_FILE = 'statefile'
KEY_GL_COLORMAP = 'glclient/colormap'
KEY_GL_PORT = 'glclient/port'
KEY_GL_CLIENT_EXECUTABLE = 'glclient/executable'
KEY_GL_BACKGROUND_COLOR = 'glclient/bgcolor'

KEY_WINDOW_GEOMETRY = 'main/geometry'
KEY_WINDOW_LAYOUT = 'main/layout'
KEY_RUNTIME_AUTOHIDE = 'main/rtautohide'
KEY_DEMOS_DIR = 'main/demosdir'

QT_VERSION = str(QtCore.QT_VERSION_STR).split('.')
QT_MAJOR_VERSION = int(QT_VERSION[0])
QT_MINOR_VERSION = int(QT_VERSION[1])

def get_settings():
    '''Initializes the QSettings for the application and returns it.'''
    global settings
    if not settings:
	QtCore.QCoreApplication.setOrganizationName('NCBS')
	QtCore.QCoreApplication.setOrganizationDomain('ncbs.res.in')
	QtCore.QCoreApplication.setApplicationName('MOOSE')
    settings = QtCore.QSettings()
    return settings

LOG_FILENAME = 'moose.log'
LOG_LEVEL = logging.DEBUG
logging.basicConfig(filename=LOG_FILENAME, level=LOG_LEVEL, filemode='w', format='%(asctime)s %(levelname)s %(name)s %(filename)s %(funcName)s: %(lineno)d: %(message)s')
LOGGER = logging.getLogger('moose')
BENCHMARK_LOGGER = logging.getLogger('moose.benchmark')
BENCHMARK_LOGGER.setLevel(logging.INFO)

# 
# config.py ends here
