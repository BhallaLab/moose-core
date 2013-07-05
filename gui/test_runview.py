# test_runview.py --- 
# 
# Filename: test_runview.py
# Description: 
# Author: 
# Maintainer: 
# Created: Wed Jul  3 10:40:05 2013 (+0530)
# Version: 
# Last-Updated: Wed Jul  3 11:08:16 2013 (+0530)
#           By: subha
#     Update #: 40
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
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt
from PyQt4.QtTest import QTest
import unittest

from mgui import MWindow
from mload import loadFile

def main():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(SaverTestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)

class RunViewKkitTestCase(unittest.TestCase):
    def setUp(self):
        self.app = QtGui.QApplication(sys.argv)
        self.win = MWindow()
        # self.loadKkitModel()
        # self.win.openRunView()
        # self.runView = self.win.plugin.getCurrentView()
        # self.schedulingWidget = self.runView.getSchedulingDockWidget().widget()


    def loadKkitModel(self):
        """Replicates the results of loading file via loadModelDialogSlot in
        MWindow"""
        loadFile('../Demose/Genesis_files/Kholodenko.g', '/model/Kholodenko', merge=False)
        self.win.setPlugin('kkit', '/model/Kholodenko')
        
    def testResetAndRun(self):
        self.schedulingWidget.resetAndRunSlot()


if __name__ == '__main__':
    unittest.main()



# 
# test_runview.py ends here
