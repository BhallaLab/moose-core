# mooseshell.py --- 
# 
# Filename: mooseshell.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Sat Jan 30 18:56:46 2010 (+0530)
# Version: 
# Last-Updated: Wed Oct 12 14:11:11 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 149
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

import sys
import code
from PyCute import PyCute

from mooseglobals import MooseGlobals
import config
from moose import *

class MooseShell(PyCute):
    """An interactive shell interface for MOOSE.

    This can either interact directly with the Python interpreter, or
    act like GENESIS command line."""
    def __init__(self, interpreter=None, mode=MooseGlobals.CMD_MODE_PYMOOSE, message='', log='', parent=None):
        if interpreter is None:
            interpreter = code.InteractiveInterpreter()
        PyCute.__init__(self, interpreter, message=message, log=log, parent=parent)
        self.mode = mode
        self.py_ps1 = sys.ps1
        self.py_ps2 = sys.ps2
        self.gen_ps1 = 'MOOSE > '
        self.gen_ps2 = '        '
        self.interpreter.runsource('from __main__ import *')
        self.interpreter.runsource('from moose import *')
        self.mooseContext = PyMooseBase.getContext()
        self.setLineWrapMode(self.WidgetWidth)

    def setMode(self, mode=MooseGlobals.CMD_MODE_PYMOOSE):
        """Toggle current command mode"""
        if mode == MooseGlobals.CMD_MODE_GENESIS:
            sys.ps1 = self.gen_ps1
            sys.ps2 = self.gen_ps2
        elif mode == MooseGlobals.CMD_MODE_PYMOOSE:
            sys.ps1 = self.py_ps1
            sys.ps2 = self.py_ps2
        else:
            config.LOGGER.error('Unknown mode ' + mode + ' specified')
            return
        self.mode = mode
        self.write(sys.ps1)

    def _run(self):
        """Override _run method in PyCute. 

        This is because we need to do different things depending on
        the mode.

        """
        self.pointer = 0
        self.history.append(QtCore.QString(self.line))
        try:
            self.lines.append(str(self.line))
        except Exception,e:
            print e
        if self.mode == MooseGlobals.CMD_MODE_PYMOOSE:
            source = '\n'.join(self.lines)
        else:
            source = 'moose.PyMooseBase.getContext().runG("' + '\n'.join(self.lines) + '")'
            
        self.more = self.interpreter.runsource(source)

        config.LOGGER.debug(self.more)
        if self.more:
            self.write(sys.ps2)
        else:
            self.write(sys.ps1)
            self.lines = []
        self._clearLine()

from PyQt4 import QtGui, QtCore
from PyQt4.Qt import Qt
        
if __name__ == '__main__':
    import sys
    import code
    app =QtGui.QApplication(sys.argv)
    interpreter = code.InteractiveInterpreter(locals=locals) 
    shell = MooseShell(interpreter)
    shell.show()
    app.exec_()
    
# 
# mooseshell.py ends here
