# firsttime.py --- 
# 
# Filename: firsttime.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Sun Jul 11 15:31:00 2010 (+0530)
# Version: 
# Last-Updated: Fri Sep 21 17:17:06 2012 (+0530)
#           By: subha
#     Update #: 414
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

import shutil
import os
from PyQt4 import QtGui, QtCore
import config
from mooseglobals import MooseGlobals

def init_config():
    """Check if there is a `.moose` directory in user's home
    directory. If not, we assume this to be the first run of MOOSE and
    copy the Demos to users home diretory."""
    moose_cfg_dir = os.path.join(os.environ['HOME'], '.moose')
    moose_local_dir = os.path.join(os.environ['HOME'], 'moose')
    if not os.path.exists(moose_cfg_dir):
        os.mkdir(moose_cfg_dir)
        print 'Created moose configuration directory:', moose_cfg_dir
    if not os.path.exists(moose_local_dir):
        os.mkdir(moose_local_dir)
        print 'Created local moose directory:', moose_local_dir    
    config.get_settings().setValue(config.KEY_LOCAL_DEMOS_DIR, os.path.join(moose_local_dir, 'Demos'))
    if os.name != 'posix':
        print """I need a posix system to automatically copy the demos
        to your home directory. Your system is:%s.To play with the
        demos you can copy the MOOSE Demos directory manually to a
        location of your choice""" % (os.name)
        return False
    moose_demos_dir = config.MOOSE_DEMOS_DIR
    if not os.access(moose_demos_dir, os.R_OK + os.X_OK):
        # Is it built from source? Then Demos directory will be
        # located in the parent directory.
        moose_demos_dir = os.path.normpath(os.path.join(config.MOOSE_GUI_DIR, '../Demos'))
        if not os.access(moose_demos_dir, os.R_OK + os.X_OK):
            print """Could not access Demos directory: %s
They will not be copied.""" % (moose_demos_dir)
            return False
        else:
            config.MOOSE_DEMOS_DIR = moose_demos_dir        
    # try:
    #     localdemosdir = os.path.join(moose_local_dir, 'Demos')
    #     print 'Copying Demos to %s' % (localdemosdir)
    #     shutil.copytree(moose_demos_dir, localdemosdir)
    #     print 'Successfully copied moose demos to:%s/Demos' % (moose_local_dir)
    # except Exception, e:
    #     print e
    #     return False
    return True


# class FileCopyDialog(QtGui.QDialog):
#     def __init__(self, *args):
#         QtGui.QDialog.__init__(self, *args)    
#         self.progressDialog = 
#         # self.setAutoClose(True)
#         # moose_demos_dir = config.get_settings.getValue(config.KEY_DEMOS_DIR)
#         # if not os.access(moose_demos_dir, os.R_OK + os.X_OK):
#         #     # Is it built from source? Then Demos directory will be
#         #     # located in the parent directory.
#         #     moose_demos_dir = os.path.join(config.MOOSE_GUI_DIR, '../Demos')
#         #     if not os.access(moose_demos_dir, os.R_OK + os.X_OK):
#         #         QtGui.QMessageBox.warning('Error', 'Could not access moose Demos directory.')
#         #         return
#         #     else:
#         #         config.MOOSE_DEMOS_DIR = moose_demos_dir        
#         #         config.get_settings().setValue(config.KEY_DEMOS_DIR, moose_demos_dir)

def copyTree(src, dst, progressDialog):
    """Copy the contents of source directory recursively into
    destination directory recursively."""
    src = src.strip()
    dst = dst.strip()
    errors = []
    if not os.access(src, os.R_OK + os.X_OK):
        print 'Failed to access directory', src
        return
    print 'Copying %s to : %s'  % (src, dst)
    try:
        os.makedirs(dst)
    except OSError, e:
        print e
        errors.append(e)
    size = 0
    for dirpath, dirnames, filenames in os.walk(src):
        for fname in filenames:
            srcname = os.path.join(dirpath, fname)
            try:
                size += os.path.getsize(srcname)
            except OSError, e:
                print e
                errors.append(e)
    progressDialog.setMaximum(size)
    print 'Total size to copy', size
    size = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dstdir = os.path.join(dst, os.path.split(dirpath)[-1])
        try:
            os.makedirs(dstdir)
        except OSError, e:
            print e
            errors.append(e)
        print 'Copying files from %s to %s' % (dirpath, dstdir)
        for fname in filenames:
            srcname = os.path.join(dirpath, fname)
            dstname = os.path.join(dstdir, fname)
            print 'Copying:', srcname, 'to', dstname        
            try:
                shutil.copy2(srcname, dstname)
            except IOError, e:
                print e
                errors.append(e)
            size += os.path.getsize(srcname)
            progressDialog.setValue(size)            
            if progressDialog.wasCanceled():
                return errors
            print 'Copied till:', size
    progressDialog.close()
    return errors
        
class ConfigWizard(QtGui.QWizard):
    """Wizard to walk users through the process of running moosegui
    for the first time."""
    def __init__(self, parent=None):
        init_config()
        QtGui.QWizard.__init__(self, parent)
        self._demosDir = str(config.get_settings().value(config.KEY_DEMOS_DIR).toString())
        if not self._demosDir:
            self._demosDir = config.MOOSE_DEMOS_DIR
        self._colormapPath = str(config.get_settings().value(config.KEY_COLORMAP_DIR).toString())
        if not self._colormapPath:
            self._colormapPath = os.path.join(config.MOOSE_GUI_DIR, 'colormaps')
        self.addPage(self._createIntroPage())
        self.addPage(self._createDemosPage())
        #self.addPage(self._createGLClientPage())
        self.addPage(self._createColormapPage())
        self.connect(self, QtCore.SIGNAL('accepted()'), self._finished)

    def _createIntroPage(self):
        page = QtGui.QWizardPage(self)
        page.setTitle('Before we start, let us verify the settings. Click Next to continue.')
        page.setSubTitle('In the next few pages you can select custom locations for the <i>demos directory</i> and <i>colormap</i> file for visualization.')
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
        page.setTitle('Select the directory containing the demos.')
        page.setSubTitle('It is normally %s. Click the browse button and select another location if you have it somewhere else.' % (self._demosDir))
        label = QtGui.QLabel('Demos directory:', page)
        line = QtGui.QLineEdit(self._demosDir, page)
        button = QtGui.QPushButton(self.tr('Browse'), page)
        self.connect(button, QtCore.SIGNAL('clicked()'), self._locateDemosDir)
        info = QtGui.QLabel("""When you finish this I\'ll try to copy the demos to your home directory. 
Don\'t worry: nothing will be overwritten.""")
        layout = QtGui.QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(line)
        layout.addWidget(button)
        panel = QtGui.QFrame()
        panel.setLayout(layout)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(panel)
        layout.addWidget(info)
        page.setLayout(layout)
        page.registerField('demosdir', line)
        self._demosDirLine = line
        return page

    def _createColormapPage(self):
        page = QtGui.QWizardPage(self)
        page.setTitle('Select the colormaps folder to use for visualization.')
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
        self._demosDir = unicode(QtGui.QFileDialog.getExistingDirectory(self, 'Demos directory'))
        self._demosDirLine.setText(self._demosDir)

    def _locateColormapFile(self):
        self._colormapPath = unicode(QtGui.QFileDialog.getExistingDirectory(self, 'Colormaps directory'))
        self._colormapLine.setText(self._colormapPath)

    def _finished(self):
        config.get_settings().setValue(config.KEY_FIRSTTIME, QtCore.QVariant(False))
        config.get_settings().setValue(config.KEY_DEMOS_DIR, self.field('demosdir'))
        print self.field('demosdir').toString()
        config.get_settings().setValue(config.KEY_COLORMAP_DIR, self.field('colormap'))
        config.get_settings().setValue(config.KEY_LOCAL_DEMOS_DIR, os.path.join(config.MOOSE_LOCAL_DIR, 'Demos'))
	config.get_settings().sync()              
        progressDialog = QtGui.QProgressDialog(self)
        progressDialog.setLabelText('Copying Demos to your home directory')
        progressDialog.setAutoClose(True)
        progressDialog.show()
        copyTree(str(self.field('demosdir').toString()), '/tmp/Demos', progressDialog)
        
if __name__ == '__main__':
    app =QtGui.QApplication([])
    widget = ConfigWizard()
    widget.show()
    app.exec_()
        

