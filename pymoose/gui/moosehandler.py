# moosehandler.py --- 
# 
# Filename: moosehandler.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Tue Jun 16 12:25:40 2009 (+0530)
# Version: 
# Last-Updated: Wed Jul  1 12:55:23 2009 (+0530)
#           By: subhasis ray
#     Update #: 150
# URL: 
# Keywords: 
# Compatibility: 
# 
# 


# Commentary: 
# 
# This is the MOOSE handler object for interaction with the GUI
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
sys.path.append('..')
import os

from PyQt4 import QtCore

import moose

class MHandler(QtCore.QThread):
    file_types = {
        'Genesis Script(*.g)':'GENESIS',
        'SBML(*.xml *.bz2 *.zip *.gz)':'SBML',
        'MOOSE(*.py)':'MOOSE'
        }
    def __init__(self, *args):
        QtCore.QObject.__init__(self, *args)
	self.context = moose.PyMooseBase.getContext()
	self.root = moose.Neutral('/')
	self.lib = moose.Neutral('/library')
	self.data = moose.Neutral('/data')
	self.proto = moose.Neutral('/proto')
        self.runTime = 1e-2 # default value
        self.updateInterval = 100 # stepsdefault value
        self.stop_ = False

    def load(self, fileName, fileType):
        """Load a file of specified type and add the directory in search path"""
        fileName = str(fileName)
        directory = os.path.dirname(fileName)
        fileType = str(fileType)
        if fileType == 'GENESIS':
            fileName = os.path.basename(fileName)
            moose.Property.addSimPath(directory)
            self.context.loadG(fileName)
        elif fileType == 'SBML':
            self.context.runG('readSBML ' + fileName + ' /kinetics')
        elif fileType == 'MOOSE':
            import subprocess
            subprocess.call(['python', fileName])
        else:
            print 'MHandler.load(): Unknown file type'
            return False

        return True
    
    def reset(self):
        print 'reset'
        self.context.reset()

    
    def run(self):
        print self.__class__.__name__,':run'
        lastTime = self.currentTime()
        print 'runtime:', self.runTime, 'update steps:', self.updateInterval
        while self.currentTime() - lastTime < self.runTime and not self.stop_:
            self.context.step(int(self.updateInterval))
            self.emit(QtCore.SIGNAL('updated()'))

    def stop(self):
        print 'Stopping'
	self.stop_ = True

    def getDataObjects(self):
        return [moose.Table(table) for table in self.data.children()]

    def getKKitGraphs(self):
        conc_list = []
        if self.context.exists('/graphs'):
            for conc_id in moose.Neutral('/graphs').children():
                conc = moose.Neutral(conc_id)
                conc_list.append(conc)
        if self.context.exists('/moregraphs'):
            for conc_id in moose.Neutral('/moregraphs').children():
                conc = moose.Neutral(conc_id)
                conc_list.append(conc)
        return conc_list

    def currentTime(self):
        return self.context.getCurrentTime()

    def getDt(self, mooseObject):
        ret = -1
        clocks = mooseObject.neighbours('process')
        if len(clocks) == 1:
            clock = moose.Tick(clocks[0])
            ret = clock.dt
        return ret

# 
# moosehandler.py ends here
