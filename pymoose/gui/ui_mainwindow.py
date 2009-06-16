# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'moose_gui.ui'
#
# Created: Wed Jun 17 01:20:41 2009
#      by: PyQt4 UI code generator 4.3.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(QtCore.QSize(QtCore.QRect(0,0,800,600).size()).expandedTo(MainWindow.minimumSizeHint()))

        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setGeometry(QtCore.QRect(0,25,800,550))
        self.centralwidget.setObjectName("centralwidget")

        self.widget = QtGui.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(-1,-1,801,571))
        self.widget.setObjectName("widget")

        self.tabWidget = QtGui.QTabWidget(self.widget)
        self.tabWidget.setGeometry(QtCore.QRect(4,-1,801,551))
        self.tabWidget.setObjectName("tabWidget")

        self.tab = QtGui.QWidget()
        self.tab.setGeometry(QtCore.QRect(0,0,797,521))
        self.tab.setObjectName("tab")
        self.tabWidget.addTab(self.tab,"")

        self.tab_2 = QtGui.QWidget()
        self.tab_2.setGeometry(QtCore.QRect(0,0,797,521))
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2,"")
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0,0,800,25))
        self.menubar.setObjectName("menubar")

        self.menuMOOSE = QtGui.QMenu(self.menubar)
        self.menuMOOSE.setObjectName("menuMOOSE")

        self.menuTutorials = QtGui.QMenu(self.menubar)
        self.menuTutorials.setObjectName("menuTutorials")

        self.menuHelp = QtGui.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")

        self.menuRun = QtGui.QMenu(self.menubar)
        self.menuRun.setObjectName("menuRun")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setGeometry(QtCore.QRect(0,575,800,25))
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.actionLoad = QtGui.QAction(MainWindow)
        self.actionLoad.setObjectName("actionLoad")

        self.actionDocumentation = QtGui.QAction(MainWindow)
        self.actionDocumentation.setObjectName("actionDocumentation")

        self.actionAbout_MOOSE = QtGui.QAction(MainWindow)
        self.actionAbout_MOOSE.setObjectName("actionAbout_MOOSE")

        self.actionReset = QtGui.QAction(MainWindow)
        self.actionReset.setObjectName("actionReset")

        self.actionStart = QtGui.QAction(MainWindow)
        self.actionStart.setObjectName("actionStart")

        self.actionStop = QtGui.QAction(MainWindow)
        self.actionStop.setObjectName("actionStop")

        self.actionQuit = QtGui.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")

        self.actionSquid_Axon = QtGui.QAction(MainWindow)
        self.actionSquid_Axon.setObjectName("actionSquid_Axon")

        self.actionIzhikevich_Neurons = QtGui.QAction(MainWindow)
        self.actionIzhikevich_Neurons.setObjectName("actionIzhikevich_Neurons")
        self.menuMOOSE.addAction(self.actionLoad)
        self.menuMOOSE.addAction(self.actionQuit)
        self.menuTutorials.addAction(self.actionSquid_Axon)
        self.menuTutorials.addAction(self.actionIzhikevich_Neurons)
        self.menuHelp.addAction(self.actionDocumentation)
        self.menuHelp.addAction(self.actionAbout_MOOSE)
        self.menuRun.addAction(self.actionReset)
        self.menuRun.addAction(self.actionStart)
        self.menuRun.addAction(self.actionStop)
        self.menubar.addAction(self.menuMOOSE.menuAction())
        self.menubar.addAction(self.menuRun.menuAction())
        self.menubar.addAction(self.menuTutorials.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QtGui.QApplication.translate("MainWindow", "Tab 1", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QtGui.QApplication.translate("MainWindow", "Tab 2", None, QtGui.QApplication.UnicodeUTF8))
        self.menuMOOSE.setTitle(QtGui.QApplication.translate("MainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.menuTutorials.setTitle(QtGui.QApplication.translate("MainWindow", "Tutorials", None, QtGui.QApplication.UnicodeUTF8))
        self.menuHelp.setTitle(QtGui.QApplication.translate("MainWindow", "Help", None, QtGui.QApplication.UnicodeUTF8))
        self.menuRun.setTitle(QtGui.QApplication.translate("MainWindow", "Run", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad.setText(QtGui.QApplication.translate("MainWindow", "Load", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad.setToolTip(QtGui.QApplication.translate("MainWindow", "Load a MOOSE/GENESIS/SBML Model", None, QtGui.QApplication.UnicodeUTF8))
        self.actionLoad.setShortcut(QtGui.QApplication.translate("MainWindow", "Ctrl+L", None, QtGui.QApplication.UnicodeUTF8))
        self.actionDocumentation.setText(QtGui.QApplication.translate("MainWindow", "Documentation", None, QtGui.QApplication.UnicodeUTF8))
        self.actionDocumentation.setShortcut(QtGui.QApplication.translate("MainWindow", "F1", None, QtGui.QApplication.UnicodeUTF8))
        self.actionAbout_MOOSE.setText(QtGui.QApplication.translate("MainWindow", "About MOOSE", None, QtGui.QApplication.UnicodeUTF8))
        self.actionReset.setText(QtGui.QApplication.translate("MainWindow", "Reset", None, QtGui.QApplication.UnicodeUTF8))
        self.actionReset.setShortcut(QtGui.QApplication.translate("MainWindow", "F7", None, QtGui.QApplication.UnicodeUTF8))
        self.actionStart.setText(QtGui.QApplication.translate("MainWindow", "Start", None, QtGui.QApplication.UnicodeUTF8))
        self.actionStart.setShortcut(QtGui.QApplication.translate("MainWindow", "F5", None, QtGui.QApplication.UnicodeUTF8))
        self.actionStop.setText(QtGui.QApplication.translate("MainWindow", "Stop", None, QtGui.QApplication.UnicodeUTF8))
        self.actionStop.setShortcut(QtGui.QApplication.translate("MainWindow", "F6", None, QtGui.QApplication.UnicodeUTF8))
        self.actionQuit.setText(QtGui.QApplication.translate("MainWindow", "Quit", None, QtGui.QApplication.UnicodeUTF8))
        self.actionQuit.setShortcut(QtGui.QApplication.translate("MainWindow", "Ctrl+Q", None, QtGui.QApplication.UnicodeUTF8))
        self.actionSquid_Axon.setText(QtGui.QApplication.translate("MainWindow", "Squid Axon", None, QtGui.QApplication.UnicodeUTF8))
        self.actionIzhikevich_Neurons.setText(QtGui.QApplication.translate("MainWindow", "Izhikevich Neurons", None, QtGui.QApplication.UnicodeUTF8))

