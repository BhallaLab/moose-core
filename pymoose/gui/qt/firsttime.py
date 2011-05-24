# firsttime.py --- 
# 
# Filename: firsttime.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Sun Jul 11 15:31:00 2010 (+0530)
# Version: 
# Last-Updated: Wed Nov 10 07:11:40 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 153
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Wizard to take the user through selection of some basic
# configurations for MOOSE gui.
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

from PyQt4 import QtGui, QtCore
import config
from mooseglobals import MooseGlobals

class FirstTimeWizard(QtGui.QWizard):
    """Wizard to walk users through the process of running moosegui
    for the first time."""
    def __init__(self, parent=None):
        QtGui.QWizard.__init__(self, parent)
        self._pymooseDemosDir = str(config.get_settings().value(config.KEY_DEMOS_DIR).toString())
        if not self._pymooseDemosDir:
            self._pymooseDemosDir = '/usr/share/doc/moose1.3/DEMOS/pymoose'
        self._glclientPath = str(config.get_settings().value(config.KEY_GL_CLIENT_EXECUTABLE).toString())
        if not self._glclientPath:
            self._glclientPath = '/usr/bin/glclient'
        self._colormapPath = str(config.get_settings().value(config.KEY_GL_COLORMAP).toString())
        if not self._colormapPath:
            self._colormapPath = '/usr/share/moose1.3/colormaps/rainbow2'
        self.addPage(self._createIntroPage())
        self.addPage(self._createDemosPage())
        self.addPage(self._createGLClientPage())
        self.addPage(self._createColormapPage())
        self.connect(self, QtCore.SIGNAL('accepted()'), self._finished)

    def _createIntroPage(self):
        page = QtGui.QWizardPage(self)
        page.setTitle('Before we start, let us verify the initial settings. Click Next to continue.')
        page.setSubTitle('In the next three pages you can select custom locations for the <i>pymoose demos directory</i>, <i>glclient</i> executable and <i>colormap</i> file for 3D visualization.')
        label = QtGui.QLabel(self.tr('<html><h3 align="center">%s</h3><p>%s</p><p>%s</p><p>%s</p><p align="center">Home Page: <a href="%s">%s</a></p></html>' % \
                (MooseGlobals.TITLE_TEXT, 
                 MooseGlobals.COPYRIGHT_TEXT, 
                 MooseGlobals.LICENSE_TEXT, 
                 MooseGlobals.ABOUT_TEXT,
                 MooseGlobals.WEBSITE,
                 MooseGlobals.WEBSITE)), page)
        label.setAlignment(QtCore.Qt.AlignHCenter)
        label.setWordWrap(True)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(label)
        page.setLayout(layout)
        return page

    def _createDemosPage(self):
        page = QtGui.QWizardPage(self)
        page.setTitle('Select the directory containing the pymoose demos.')
        page.setSubTitle('It is normally %s. Click the browse button and select another location if you have it somewhere else.' % self._pymooseDemosDir)
        label = QtGui.QLabel('PyMOOSE demos directory:', page)
        line = QtGui.QLineEdit(self._pymooseDemosDir, page)
        button = QtGui.QPushButton(self.tr('Browse'), page)
        self.connect(button, QtCore.SIGNAL('clicked()'), self._locateDemosDir)
        layout = QtGui.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(line)
        layout.addWidget(button)
        page.setLayout(layout)
        page.registerField('demosdir', line)
        self._demosDirLine = line
        return page
        
    def _createGLClientPage(self):
        page = QtGui.QWizardPage(self)
        page.setTitle('Select the glclient executable file.')
        page.setSubTitle('The program for 3D visualization in MOOSE is called glclient.\nThe glclient file is installed in /usr/bin\nBut if you have it somewhere else, please select it below.' )
        label = QtGui.QLabel('glclient executable', page)
        line = QtGui.QLineEdit(self._glclientPath, page)
        button = QtGui.QPushButton(self.tr('Browse'), page)
        self.connect(button, QtCore.SIGNAL('clicked()'), self._locateGLClient)
        layout = QtGui.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(line)
        layout.addWidget(button)
        page.setLayout(layout)
        page.registerField('glclient', line)
        self._glclientLine = line
        return page

    def _createColormapPage(self):
        page = QtGui.QWizardPage(self)
        page.setTitle('Select the colormaps folder to use for 3D visualization.')
        page.setSubTitle('The colormap folder is %s\nBut if you have it somewhere else, please select it below.' % (self._colormapPath))
        label = QtGui.QLabel('Colormap file', page)
        line = QtGui.QLineEdit(self._colormapPath, page)
        button = QtGui.QPushButton(self.tr('Browse'), page)
        self.connect(button, QtCore.SIGNAL('clicked()'), self._locateColormapFile)
        layout = QtGui.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(line)
        layout.addWidget(button)
        page.setLayout(layout)
        page.registerField('colormap', line)
        self._colormapLine = line
        return page 

    def _locateDemosDir(self):
        self._pymooseDemosDir = unicode(QtGui.QFileDialog.getExistingDirectory(self, 'PyMOOSE Demos Directory'))
        self._demosDirLine.setText(self._pymooseDemosDir)

    def _locateGLClient(self):
        self._glclientPath = unicode(QtGui.QFileDialog.getOpenFileName(self, 'glclient program'))
        self._glclientLine.setText(self._glclientPath)

    def _locateColormapFile(self):
#        self._colormapPath = unicode(QtGui.QFileDialog.getOpenFileName(self, 'Colormap file'))
        self._colormapPath = unicode(QtGui.QFileDialog.getExistingDirectory(self, 'Colormap file'))
        self._colormapLine.setText(self._colormapPath)

    def _finished(self):
        config.get_settings().setValue(config.KEY_FIRSTTIME, QtCore.QVariant(False))
        config.get_settings().setValue(config.KEY_GL_CLIENT_EXECUTABLE, self.field('glclient'))
        config.get_settings().setValue(config.KEY_DEMOS_DIR, self.field('demosdir'))
        config.get_settings().setValue(config.KEY_GL_COLORMAP, self.field('colormap'))
	config.get_settings().sync()      
        
        
if __name__ == '__main__':
    app =QtGui.QApplication([])
    widget = FirstTimeWizard()
    widget.show()
    app.exec_()
        


# 
# firsttime.py ends here
