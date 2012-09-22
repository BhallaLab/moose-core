# config.py --- 
# 
# Filename: config.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Sat Feb 13 16:07:56 2010 (+0530)
# Version: 
# Last-Updated: Sat Sep 22 18:03:25 2012 (+0530)
#           By: subha
#     Update #: 300
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Provides keys for accessing per-user settings.
# Provides initialization of several per-user variables for MooseGUI.
# As part of initialization, creates `~/.moose` and `~/moose`
# directories.
#
# 

# Change log:
# 
# 2012-09-22 13:49:36 (+0530) Subha: cleaned up the initialization
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

TEMPDIR = tempfile.gettempdir()
KEY_FIRSTTIME = 'firsttime'
# KEY_STATE_FILE = 'statefile'

KEY_WINDOW_GEOMETRY = 'main/geometry'
KEY_WINDOW_LAYOUT = 'main/layout'
KEY_RUNTIME_AUTOHIDE = 'main/rtautohide'
KEY_DEMOS_DIR = 'main/demosdir'
KEY_DOCS_DIR = 'main/docsdir'
KEY_HOME_DIR = 'main/homedir' 
KEY_ICON_DIR = 'main/icondir' 
KEY_COLORMAP_DIR = 'main/colormapdir' 
KEY_LOCAL_DEMOS_DIR = 'main/localdemosdir'
KEY_MOOSE_LOCAL_DIR = 'main/localdir'
KEY_NUMPTHREADS = 'main/numpthreads'

QT_VERSION = str(QtCore.QT_VERSION_STR).split('.')
QT_MAJOR_VERSION = int(QT_VERSION[0])
QT_MINOR_VERSION = int(QT_VERSION[1])

MOOSE_DOC_URL = 'http://moose.ncbs.res.in/content/view/5/6/'
MOOSE_REPORT_BUG_URL = 'http://sourceforge.net/tracker/?func=add&group_id=165660&atid=836272'

MOOSE_DEMOS_DIR = '/usr/share/moose/Demos'
MOOSE_DOCS_DIR =  '/usr/share/doc/moose'
MOOSE_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
MOOSE_CFG_DIR = os.path.join(os.environ['HOME'], '.moose')
MOOSE_LOCAL_DIR = os.path.join(os.environ['HOME'], 'moose')
MOOSE_NUMPTHREADS = '1'
MOOSE_ABOUT_FILE = os.path.join(MOOSE_GUI_DIR, 'about.html')

class MooseSetting(dict):
    """
    dict-like access to QSettings.

    This subclass of dict wraps a QSettings object and lets one set
    and get values as Python strings rather than QVariant.
    
    This is supposed yo be a singleton in the whole application.
    """
    _instance = None    
    def __new__(cls, *args, **kwargs):
        # This is designed to check if the class has been
        # instantiated, if so, returns the single instance, otherwise
        # creates it.
        if cls._instance is None:           
            cls._instance = super(MooseSetting, cls).__new__(cls, *args, **kwargs)
            firsttime, errs = init_dirs()
            for e in errs:
                print e
            QtCore.QCoreApplication.setOrganizationName('NCBS')
            QtCore.QCoreApplication.setOrganizationDomain('ncbs.res.in')
            QtCore.QCoreApplication.setApplicationName('MOOSE')
            cls._instance.qsettings = QtCore.QSettings()
            # If this is the first time, then set some defaults
            if firsttime:
                cls._instance.qsettings.setValue(KEY_FIRSTTIME, False)
                cls._instance.qsettings.setValue(KEY_DEMOS_DIR, MOOSE_DEMOS_DIR)
                cls._instance.qsettings.setValue(KEY_LOCAL_DEMOS_DIR, os.path.join(MOOSE_LOCAL_DIR, 'Demos'))
                cls._instance.qsettings.setValue(KEY_DOCS_DIR, MOOSE_DOCS_DIR)
                cls._instance.qsettings.setValue(KEY_MOOSE_LOCAL_DIR, MOOSE_LOCAL_DIR)
                cls._instance.qsettings.setValue(KEY_COLORMAP_DIR, os.path.join(MOOSE_GUI_DIR, 'colormaps'))
                cls._instance.qsettings.setValue(KEY_HOME_DIR, os.environ['HOME'])
                cls._instance.qsettings.setValue(KEY_ICON_DIR, os.path.join(MOOSE_GUI_DIR, 'icons'))
                cls._instance.qsettings.setValue(KEY_NUMPTHREADS, '1')
                cls._instance.qsettings.setValue(KEY_FIRSTTIME, True)
        return cls._instance

    def __init__(self, *args, **kwargs):
        super(MooseSetting, self).__init__(self, *args, **kwargs)

    def __iter__(self):
        return (str(key) for key in self.qsettings.allKeys())
        
    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.qsettings.setValue(key, value)
        else:
            raise TypeError('Expect only strings as keys')

    def __getitem__(self, key):
        return str(self.qsettings.value(key).toString())

    def keys(self):
        return [str(key) for key in self.qsettings.allKeys()]

    def values(self):
        return [str(self.qsettings.value(key).toString()) for key in self.qsettings.allKeys()]

    def itervalues(self):
        return (str(self.qsettings.value(key).toString()) for key in self.qsettings.allKeys())

def init_dirs():
    """Check if there is a `.moose` directory in user's home
    directory. If not, we assume this to be the first run of MOOSE.
    Then we try to create the `~/.moose` directory and `~/moose`
    directory.
    """  
    firsttime = False
    global MOOSE_DEMOS_DIR
    global MOOSE_LOCAL_DIR
    global MOOSE_CFG_DIR
    global MOOSE_DOCS_DIR
    errors = []
    moose_cfg_dir = os.path.join(os.environ['HOME'], '.moose')
    moose_local_dir = os.path.join(os.environ['HOME'], 'moose')
    if not os.path.exists(moose_cfg_dir):
        firsttime = True
        try:
            os.mkdir(moose_cfg_dir)
            MOOSE_CFG_DIR = moose_cfg_dir
            print 'Created moose configuration directory:', moose_cfg_dir
        except OSError, e:
            errors.append(e)
            print e
    if not os.path.exists(moose_local_dir):
        try:
            os.mkdir(moose_local_dir)
            MOOSE_LOCAL_DIR = moose_local_dir
            print 'Created local moose directory:', moose_local_dir    
        except OSError, e:
            errors.append(e)
            print e
    moose_demos_dir = MOOSE_DEMOS_DIR
    if not os.access(moose_demos_dir, os.R_OK + os.X_OK):
        # Is it built from source? Then Demos directory will be
        # located in the parent directory.
        moose_demos_dir = os.path.normpath(os.path.join(MOOSE_GUI_DIR, '../Demos'))
        if not os.access(moose_demos_dir, os.R_OK + os.X_OK):
            print "Could not access Demos directory: %s" % (moose_demos_dir)
            errors.append(OSError(errno.EACCES, 'Cannot access %s' % (moose_demos_dir)))
        else:
            MOOSE_DEMOS_DIR = moose_demos_dir        
    moose_docs_dir = MOOSE_DOCS_DIR
    if not os.access(moose_docs_dir, os.R_OK + os.X_OK):
        # Is it built from source? Then Docs directory will be
        # located in the parent directory.
        moose_docs_dir = os.path.normpath(os.path.join(MOOSE_GUI_DIR, '../Docs'))
        if not os.access(moose_docs_dir, os.R_OK + os.X_OK):
            print "Could not access Demos directory: %s" % (moose_docs_dir)
            errors.append(OSError(errno.EACCES, 'Cannot access %s' % (moose_docs_dir)))
        else:
            MOOSE_DOCS_DIR = moose_docs_dir            
    return firsttime, errors

settings = MooseSetting()

# LOG_FILENAME = os.path.join(TEMPDIR, 'moose.log')
LOG_LEVEL = logging.ERROR
# logging.basicConfig(filename=LOG_FILENAME, level=LOG_LEVEL, filemode='w', format='%(asctime)s %(levelname)s %(name)s %(filename)s %(funcName)s: %(lineno)d: %(message)s')
logging.basicConfig(stream=sys.stdout, level=LOG_LEVEL, filemode='w', format='%(asctime)s %(levelname)s %(name)s %(filename)s %(funcName)s: %(lineno)d: %(message)s')
LOGGER = logging.getLogger('moose')
BENCHMARK_LOGGER = logging.getLogger('moose.benchmark')
BENCHMARK_LOGGER.setLevel(logging.INFO)

# 
# config.py ends here
