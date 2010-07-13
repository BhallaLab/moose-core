# glclientgui.py --- 
# 
# Filename: glclientgui.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Sat Feb 13 16:01:54 2010 (+0530)
# Version: 
# Last-Updated: Thu Jul  8 14:58:12 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 240
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# This opens a GUI to start glclient and specify the parameters.
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
import os

from PyQt4.Qt import Qt
from PyQt4 import QtGui, QtCore

import config
from glclient import GLClient

class GLClientGUI(QtGui.QWidget):
    '''This is a GUI for setting the parameters for the glclient. The
    actual client process is created only after the parameters hae
    been set and the user pushes the "Start Client" button.'''
    def __init__(self, *args):
	QtGui.QWidget.__init__(self, *args)
        self.client = None
        self.init_settings()
        
        # Create the componet widgets
        self.exeLabel = QtGui.QLabel('GL Client executable', self)
        self.exeText = QtGui.QLineEdit(self.executable, self)
        self.exeButton = QtGui.QPushButton('Open', self)
        self.modeLabel = QtGui.QLabel('Mode')
        self.modeCombo = QtGui.QComboBox(self)
        self.modeCombo.addItem('GLCell')
        self.modeCombo.addItem('GLView')
        self.portLabel = QtGui.QLabel('Port', self)
        self.portText = QtGui.QLineEdit(self.port, self)
        self.colormapLabel = QtGui.QLabel('Colormap File', self)
        self.colormapText = QtGui.QLineEdit(self.colormap, self)
        self.colormapButton = QtGui.QPushButton('Open', self)
        self.startClientButton = QtGui.QPushButton('Start GLClient', self)
        self.stopClientButton =  QtGui.QPushButton('Stop GLClient', self)
        # Add some action
        self.connect(self.exeButton, QtCore.SIGNAL('clicked()'), self.open_executable_dialog)
        self.connect(self.colormapButton, QtCore.SIGNAL('clicked()'), self.open_colormap_dialog)
        self.connect(self.startClientButton, QtCore.SIGNAL('clicked()'), self.start_client)
        self.connect(self.stopClientButton, QtCore.SIGNAL('clicked()'), self.stop_client)

        # Create the layout 
        layout = QtGui.QGridLayout(self)
        layout.addWidget(self.exeLabel, 0, 0)
        layout.addWidget(self.exeText, 0, 1)
        layout.addWidget(self.exeButton, 0, 2)
        layout.addWidget(self.colormapLabel, 1, 0)
        layout.addWidget(self.colormapText, 1, 1)
        layout.addWidget(self.colormapButton, 1, 2)
        layout.addWidget(self.portLabel, 2, 0)
        layout.addWidget(self.portText, 2, 1)
        layout.addWidget(self.modeLabel, 3, 0)
        layout.addWidget(self.modeCombo, 3, 1)
        layout.addWidget(self.startClientButton, 4, 0)
        layout.addWidget(self.stopClientButton, 4, 2)


    def open_colormap_dialog(self):
        self.colormap = unicode(QtGui.QFileDialog.getOpenFileName(self, 'Select Colormap File'))
        self.colormapText.setText(self.colormap)

    def open_executable_dialog(self):
        self.executable = unicode(QtGui.QFileDialog.getOpenFileName(self, 'Select GLClient Executable'))
        if not os.access(self.executable, os.X_OK):
            QtGui.QMessageBox.critical(self, 'GLClient executable', 'Please select the correct glclient executable. Make sure you have execute permission on this file')
        else:
            self.exeText.setText(self.executable)

    def start_client(self):
        self.settings.beginGroup('glclient')
        self.new_port = self.portText.text()
        if str(self.new_port) != str(self.port):
            self.port = self.new_port
            self.settings.setValue('port', self.port)
        self.new_colormap = unicode(self.colormapText.text())
        if self.new_colormap != self.colormap:
            self.colormap = self.new_colormap
        if not os.path.isfile(self.colormap):
            self.colormap = unicode(QtGui.QFileDialog.getOpenFileName(self, 'Select Colormap File'))
        self.settings.setValue('colormap', self.colormap)
            
        self.new_executable = unicode(self.exeText.text())
        if self.executable != self.new_executable:
            self.executable = self.new_executable
        if not os.access(self.executable, os.X_OK):
            self.executable = unicode(QtGui.QFileDialog.getOpenFileName(self, 'Select GLClient Executable'))
        if not os.access(self.executable, os.X_OK):
            QtGui.QMessageBox.critical(self, 'GLClient executable', 'Please select the correct glclient executable. Make sure you have execute permission on this file')            
        self.settings.endGroup()
        if str(self.modeCombo.currentText()) == 'GLCell':
            mode = 'c'
        elif str(self.modeCombo.currentText()) == 'GLView':
            mode = 'v'
        self.client = GLClient(exe=self.executable, port=self.port, mode=mode, colormap=self.colormap)
        
    def stop_client(self):
        self.client.stop()


    def init_settings(self):
	self.settings = config.get_settings()
        self.port = unicode(self.settings.value(config.KEY_GL_PORT).toString())
        if not self.port:
            self.port = config.GL_PORT
            self.settings.setValue(config.KEY_GL_PORT, self.port)
            config.LOGGER.info('new settings')
        print unicode(self.settings.value(config.KEY_GL_PORT).toString())
        self.colormap = unicode(self.settings.value(config.KEY_GL_COLORMAP).toString())
        if not self.colormap:
            self.colormap = config.GL_DEFAULT_COLORMAP
        if not os.path.isfile(self.colormap):
            config.LOGGER.error('Colormap file not found: %s' %( self.colormap))
            self.colormap = unicode(QtGui.QFileDialog.getOpenFileName(self, 'Select Colormap File'))
        self.settings.setValue(config.KEY_GL_COLORMAP, self.colormap)
        self.executable = unicode(self.settings.value(config.KEY_GL_CLIENT_EXECUTABLE).toString()        )
        if not self.executable:
            self.executable = config.GL_CLIENT_EXECUTABLE
        if not os.access(self.executable, os.X_OK):        
            self.executable = QtGui.QFileDialog.getOpenFileName(self, 'Select GLClient Executable')
        self.settings.setValue(config.KEY_GL_CLIENT_EXECUTABLE, self.executable)
        if not os.access(self.executable, os.X_OK):        
            QtGui.QMessageBox.critical(self, 'GLClient executable', 'Please select the correct glclient executable. Make sure you have execute permission on this file')


        
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    widget = GLClientGUI()
    widget.show()
    app.exec_()

# 
# glclientgui.py ends here
