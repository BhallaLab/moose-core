# config.py --- 
# 
# Filename: config.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Sat Feb 13 16:07:56 2010 (+0530)
# Version: 
# Last-Updated: Thu Sep 27 17:52:21 2012 (+0530)
#           By: subha
#     Update #: 346
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
## import moose just for version number
import moose
MOOSE_VERSION = moose.VERSION

TEMPDIR = tempfile.gettempdir()
KEY_FIRSTTIME = 'firsttime'
KEY_FIRSTTIME_THISVERSION = 'firsttime_'+MOOSE_VERSION
KEY_DEMOS_COPIED_THISVERSION = 'demoscopied_'+MOOSE_VERSION
# KEY_STATE_FILE = 'statefile'

TRUE_STRS = ['True', 'true', '1', 'Yes', 'yes', 'Y']

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
KEY_LOCAL_BUILD = 'main/localbuild'

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

# If we have a Makefile above GUI directory, then this must be a
# locally built version
LOCAL_BUILD = os.access(os.path.join(MOOSE_GUI_DIR, '../Makefile'), os.R_OK)

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
            QtCore.QCoreApplication.setOrganizationName('NCBS')
            QtCore.QCoreApplication.setOrganizationDomain('ncbs.res.in')
            QtCore.QCoreApplication.setApplicationName('MOOSE')
            cls._instance.qsettings = QtCore.QSettings()
            ## if this version's firstime key is not present '' or is True,
            ## then set firsttime=True and do what needs to be done first time.
            if cls._instance[KEY_FIRSTTIME_THISVERSION] in ['']+TRUE_STRS:
                firsttime = True
                errs = init_dirs()
                for e in errs:
                    print e
            else:
                firsttime = False
            # If this is the first time, then set some defaults
            if firsttime:
                cls._instance.qsettings.setValue(KEY_FIRSTTIME_THISVERSION, 'True') # string not boolean
                cls._instance.qsettings.setValue(KEY_DEMOS_COPIED_THISVERSION, 'False') # string not boolean
                cls._instance.qsettings.setValue(KEY_COLORMAP_DIR, os.path.join(MOOSE_GUI_DIR, 'colormaps'))
                cls._instance.qsettings.setValue(KEY_ICON_DIR, os.path.join(MOOSE_GUI_DIR, 'icons'))
                cls._instance.qsettings.setValue(KEY_NUMPTHREADS, '1')
            else:
                cls._instance.qsettings.setValue(KEY_FIRSTTIME_THISVERSION, 'False') # string not boolean
            # These are to be checked at every run
            cls._instance.qsettings.setValue(KEY_HOME_DIR, os.environ['HOME'])
            cls._instance.qsettings.setValue(KEY_DEMOS_DIR, MOOSE_DEMOS_DIR)
            cls._instance.qsettings.setValue(KEY_LOCAL_DEMOS_DIR, os.path.join(MOOSE_LOCAL_DIR, 'Demos'))
            cls._instance.qsettings.setValue(KEY_DOCS_DIR, MOOSE_DOCS_DIR)
            cls._instance.qsettings.setValue(KEY_MOOSE_LOCAL_DIR, MOOSE_LOCAL_DIR)
            cls._instance.qsettings.setValue(KEY_LOCAL_BUILD, LOCAL_BUILD)
            os.environ['NUMPTHREADS'] = str(cls._instance.qsettings.value(KEY_NUMPTHREADS).toString())
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
    """This is called during the first run of this version of MOOSE.
    We try to create the `~/moose_<version>` directory.
    """  
    global MOOSE_DEMOS_DIR
    global MOOSE_LOCAL_DIR
    global MOOSE_CFG_DIR
    global MOOSE_DOCS_DIR
    global LOCAL_BUILD
    errors = []
    if LOCAL_BUILD:
        MOOSE_LOCAL_DIR = os.path.normpath(os.path.join(MOOSE_GUI_DIR, '..'))
        MOOSE_DEMOS_DIR = os.path.join(MOOSE_LOCAL_DIR, 'Demos')
        MOOSE_DOCS_DIR = os.path.join(MOOSE_LOCAL_DIR, 'Docs')
    else:
        MOOSE_LOCAL_DIR = os.path.join(os.environ['HOME'], 'moose_'+MOOSE_VERSION)
        if not os.path.exists(MOOSE_LOCAL_DIR):
            try:
                os.mkdir(MOOSE_LOCAL_DIR)
                print 'Created local moose directory:', MOOSE_LOCAL_DIR
            except OSError, e:
                errors.append(e)
                print e
    if not os.access(MOOSE_DOCS_DIR, os.R_OK + os.X_OK):
        print "Could not access documentation directory: %s" % (MOOSE_DOCS_DIR)
        errors.append(OSError(errno.EACCES, 'Cannot access %s' % (MOOSE_DOCS_DIR)))
    return errors

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
