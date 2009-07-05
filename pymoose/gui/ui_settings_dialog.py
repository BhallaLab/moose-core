# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'settings_dialog.ui'
#
# Created: Sun Jul  5 14:43:15 2009
#      by: PyQt4 UI code generator 4.3.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_settingsDialog(object):
    def setupUi(self, settingsDialog):
        settingsDialog.setObjectName("settingsDialog")
        settingsDialog.setWindowModality(QtCore.Qt.WindowModal)
        settingsDialog.resize(QtCore.QSize(QtCore.QRect(0,0,500,300).size()).expandedTo(settingsDialog.minimumSizeHint()))

        self.vboxlayout = QtGui.QVBoxLayout(settingsDialog)
        self.vboxlayout.setObjectName("vboxlayout")

        self.gridlayout = QtGui.QGridLayout()
        self.gridlayout.setObjectName("gridlayout")

        self.label = QtGui.QLabel(settingsDialog)
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.gridlayout.addWidget(self.label,0,0,1,1)

        self.simPathTextEdit = QtGui.QTextEdit(settingsDialog)
        self.simPathTextEdit.setObjectName("simPathTextEdit")
        self.gridlayout.addWidget(self.simPathTextEdit,0,1,1,1)
        self.vboxlayout.addLayout(self.gridlayout)

        self.buttonBox = QtGui.QDialogButtonBox(settingsDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.vboxlayout.addWidget(self.buttonBox)

        self.retranslateUi(settingsDialog)
        QtCore.QObject.connect(self.buttonBox,QtCore.SIGNAL("accepted()"),settingsDialog.accept)
        QtCore.QObject.connect(self.buttonBox,QtCore.SIGNAL("rejected()"),settingsDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(settingsDialog)

    def retranslateUi(self, settingsDialog):
        settingsDialog.setWindowTitle(QtGui.QApplication.translate("settingsDialog", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("settingsDialog", "SIMPATH (Enter one directory path on each line)", None, QtGui.QApplication.UnicodeUTF8))

