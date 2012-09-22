# utils.py --- 
# 
# Filename: utils.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat Sep 22 15:19:09 2012 (+0530)
# Version: 
# Last-Updated: Sat Sep 22 15:29:59 2012 (+0530)
#           By: subha
#     Update #: 18
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

import shutil
import os
from PyQt4 import QtGui, QtCore

def resizeWidthToContents(lineEdit):
    """Resize a QLineEdit object to fit its content."""
    text = lineEdit.text()
    fm = lineEdit.fontMetrics()
    w = fm.boundingRect(text).width()
    lineEdit.resize(w, lineEdit.height())

def copyTree(src, dst, progressDialog):
    """Copy the contents of source directory recursively into
    destination directory recursively. Display the progress in
    progressDialog. This does not overwrite any existing files or
    directories.

    Parameters
    ----------
    src: str
    path of the source directory

    dst: str
    path of the destination directory

    progressDialog: QProgressDialog
    Qt object to display the progress of the copying.

    Return
    ------
    list containing all the errors encountered while copying.

    """
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



# 
# utils.py ends here
