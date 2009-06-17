# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'moose_gui.ui'
#
# Created: Wed Jun 17 23:11:13 2009
#      by: PyQt4 UI code generator 4.3.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(QtCore.QSize(QtCore.QRect(0,0,668,617).size()).expandedTo(MainWindow.minimumSizeHint()))

        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)

        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setGeometry(QtCore.QRect(0,25,668,567))
        self.centralwidget.setObjectName("centralwidget")

        self.vboxlayout = QtGui.QVBoxLayout(self.centralwidget)
        self.vboxlayout.setObjectName("vboxlayout")

        self.runControlWidget = QtGui.QWidget(self.centralwidget)

        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.runControlWidget.sizePolicy().hasHeightForWidth())
        self.runControlWidget.setSizePolicy(sizePolicy)
        self.runControlWidget.setObjectName("runControlWidget")

        self.hboxlayout = QtGui.QHBoxLayout(self.runControlWidget)
        self.hboxlayout.setObjectName("hboxlayout")

        self.resetPushButton = QtGui.QPushButton(self.runControlWidget)
        self.resetPushButton.setObjectName("resetPushButton")
        self.hboxlayout.addWidget(self.resetPushButton)

        self.runPushButton = QtGui.QPushButton(self.runControlWidget)
        self.runPushButton.setObjectName("runPushButton")
        self.hboxlayout.addWidget(self.runPushButton)

        self.stopPushButton = QtGui.QPushButton(self.runControlWidget)
        self.stopPushButton.setObjectName("stopPushButton")
        self.hboxlayout.addWidget(self.stopPushButton)

        self.runTimeLabel = QtGui.QLabel(self.runControlWidget)

        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.runTimeLabel.sizePolicy().hasHeightForWidth())
        self.runTimeLabel.setSizePolicy(sizePolicy)
        self.runTimeLabel.setObjectName("runTimeLabel")
        self.hboxlayout.addWidget(self.runTimeLabel)

        self.runTimeLineEdit = QtGui.QLineEdit(self.runControlWidget)

        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.runTimeLineEdit.sizePolicy().hasHeightForWidth())
        self.runTimeLineEdit.setSizePolicy(sizePolicy)
        self.runTimeLineEdit.setObjectName("runTimeLineEdit")
        self.hboxlayout.addWidget(self.runTimeLineEdit)
        self.vboxlayout.addWidget(self.runControlWidget)

        self.tabWidget = QtGui.QTabWidget(self.centralwidget)

        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setObjectName("tabWidget")

        self.modelTreeTab = QtGui.QWidget()
        self.modelTreeTab.setGeometry(QtCore.QRect(0,0,646,465))
        self.modelTreeTab.setObjectName("modelTreeTab")
        self.tabWidget.addTab(self.modelTreeTab,"")

        self.plotsTab = QtGui.QWidget()
        self.plotsTab.setGeometry(QtCore.QRect(0,0,1258,648))
        self.plotsTab.setObjectName("plotsTab")
        self.tabWidget.addTab(self.plotsTab,"")
        self.vboxlayout.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0,0,668,25))
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
        self.statusbar.setGeometry(QtCore.QRect(0,592,668,25))
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
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.resetPushButton.setText(QtGui.QApplication.translate("MainWindow", "Reset", None, QtGui.QApplication.UnicodeUTF8))
        self.runPushButton.setText(QtGui.QApplication.translate("MainWindow", "Run", None, QtGui.QApplication.UnicodeUTF8))
        self.stopPushButton.setText(QtGui.QApplication.translate("MainWindow", "Stop", None, QtGui.QApplication.UnicodeUTF8))
        self.runTimeLabel.setText(QtGui.QApplication.translate("MainWindow", "Run for (seconds)", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.modelTreeTab), QtGui.QApplication.translate("MainWindow", "Model Tree", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabToolTip(self.tabWidget.indexOf(self.modelTreeTab),QtGui.QApplication.translate("MainWindow", "Visualize the model as a tree structure", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.plotsTab), QtGui.QApplication.translate("MainWindow", "Plots", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabToolTip(self.tabWidget.indexOf(self.plotsTab),QtGui.QApplication.translate("MainWindow", "Display the plots from current run", None, QtGui.QApplication.UnicodeUTF8))
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

