#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NeuroKitEditor for NeuroKit plugin.
"""

__author__      =   "Aviral Goel"
__credits__     =   ["Upi Lab"]
__license__     =   "GPL3"
__version__     =   "1.0.0"
__maintainer__  =   "Aviral Goel"
__email__       =   "goel.aviral@gmail.com"
__status__      =   "Development"

import mplugin
import moose
import pprint
# import NeuroKitEditorWidget
import default

from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4 import Qt
from PyQt4.QtGui import QPushButton
from PyQt4.QtGui import QWidget
from PyQt4.QtGui import QHBoxLayout
from PyQt4.QtGui import QGridLayout
from PyQt4.QtGui import QDialog
from PyQt4.QtGui import QTableWidget
from PyQt4.QtGui import QTableWidgetItem
from PyQt4.QtGui import QCheckBox
from PyQt4.QtGui import QComboBox

class NeuroKitEditor(mplugin.EditorBase):

    """
    NeuroKitEditor
    """

    def __init__(self, plugin, modelRoot):
        super(NeuroKitEditor, self).__init__(plugin)
        self._centralWidget = default.DefaultEditorWidget(None)
        self.modelRoot = modelRoot
        # self._centralWidget = NeuroKitEditorWidget.NeuroKitEditorWidget(modelRoot)
        self._menus         = []
        # self._propertyTable = MorphologyProperyTable()
        self._propertyTable = QWidget()
        self.__initMenus()
        self.__initToolBars()
        self.setModelRoot(modelRoot)
        #     if hasattr(self._centralWidget, 'init'):
        #         self._centralWidget.init()
        #     self._centralWidget.setModelRoot(self.plugin.modelRoot)
        # return self._centralWidget

    def __initMenus(self):
        return self._menus
        # editMenu = QtGui.QMenu('&Edit')
        # for menu in self.getCentralWidget().getMenus():
        #     editMenu.addMenu(menu)
        # self._menus.append(detailsButton)

    def __initToolBars(self):
        return self._toolBars

        # for toolbar in self.getCentralWidget().getToolBars():
        #     self._toolBars.append(toolbar)

    def getToolPanes(self):
        return super(NeuroKitEditor, self).getToolPanes()

    def getLibraryPane(self):
        return super(NeuroKitEditor, self).getLibraryPane()

    def getOperationsWidget(self):
        return super(NeuroKitEditor, self).getOperationsPane()

    def getCentralWidget(self):
        """Retrieve or initialize the central widget.

        Note that we call the widget's setModelRoot() function
        explicitly with the plugin's modelRoot as the argument. This
        enforces an update of the widget display with the current
        modelRoot.

        This function should be overridden by any derived class as it
        has the editor widget class hard coded into it.
        """
        self._centralWidget.setModelRoot(self.plugin.modelRoot)
        return self._centralWidget

    def updateModelView(self):
        pass

    def setModelRoot(self, path):
        self.modelRoot = path
        self.updateModelView()
