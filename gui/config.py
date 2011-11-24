# config.py --- 
# 
# Filename: config.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Sat Feb 13 16:07:56 2010 (+0530)
# Version: 
# Last-Updated: Mon Nov 15 09:53:00 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 145
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

import os
import sys
import tempfile
import logging
from PyQt4.Qt import Qt
from PyQt4 import QtGui, QtCore

moose_version = '1.4'
settings = None
TEMPDIR = tempfile.gettempdir()
KEY_FIRSTTIME = 'firsttime'

KEY_HOME_DIR = os.path.abspath(__file__).rstrip('config.py')
KEY_MAIN_DIR = os.path.abspath(os.path.join(KEY_HOME_DIR,'..'))
KEY_ICON_DIR = os.path.join(KEY_HOME_DIR,'icons')

user_home = os.path.expanduser('~')
user_moose_dir = os.path.join(user_home, 'moose%s' % (moose_version))    
KEY_DEMOS_DIR = os.path.join(user_moose_dir,'DEMOS','pymoose')

KEY_WINDOW_GEOMETRY = os.path.join(KEY_HOME_DIR,'geometry')
KEY_WINDOW_LAYOUT = os.path.join(KEY_HOME_DIR,'layout')
KEY_RUNTIME_AUTOHIDE = os.path.join(KEY_HOME_DIR,'rtautohide')
KEY_GL_COLORMAP = os.path.join(KEY_HOME_DIR,'oglfunc','colors')

KEY_GL_BACKGROUND_COLOR = 'glclient/bgcolor'

QT_VERSION = str(QtCore.QT_VERSION_STR).split('.')
QT_MAJOR_VERSION = int(QT_VERSION[0])
QT_MINOR_VERSION = int(QT_VERSION[1])

MOOSE_DOC_FILE = os.path.abspath(os.path.join(KEY_HOME_DIR,'documentation.pdf'))
MOOSE_REPORT_BUG_URL = 'http://sourceforge.net/tracker/?func=add&group_id=165660&atid=836272'

def get_settings():
    '''Initializes the QSettings for the application and returns it.'''
    global settings
    if not settings:
	QtCore.QCoreApplication.setOrganizationName('NCBS')
	QtCore.QCoreApplication.setOrganizationDomain('ncbs.res.in')
	QtCore.QCoreApplication.setApplicationName('MOOSE')
        settings = QtCore.QSettings()
    return settings

# LOG_FILENAME = os.path.join(TEMPDIR, 'moose.log')
LOG_LEVEL = logging.ERROR
# logging.basicConfig(filename=LOG_FILENAME, level=LOG_LEVEL, filemode='w', format='%(asctime)s %(levelname)s %(name)s %(filename)s %(funcName)s: %(lineno)d: %(message)s')
logging.basicConfig(stream=sys.stdout, level=LOG_LEVEL, filemode='w', format='%(asctime)s %(levelname)s %(name)s %(filename)s %(funcName)s: %(lineno)d: %(message)s')
LOGGER = logging.getLogger('moose')
BENCHMARK_LOGGER = logging.getLogger('moose.benchmark')
BENCHMARK_LOGGER.setLevel(logging.INFO)
