# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'binProperty.ui'
#
# Created: Fri Jan  6 17:20:44 2012
#      by: PyQt4 UI code generator 4.8.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(510, 322)
        self.okayPushButton = QtGui.QPushButton(Dialog)
        self.okayPushButton.setGeometry(QtCore.QRect(390, 260, 97, 27))
        self.okayPushButton.setObjectName(_fromUtf8("okayPushButton"))
        self.noBinRadioButton = QtGui.QRadioButton(Dialog)
        self.noBinRadioButton.setGeometry(QtCore.QRect(30, 100, 281, 22))
        self.noBinRadioButton.setChecked(True)
        self.noBinRadioButton.setObjectName(_fromUtf8("noBinRadioButton"))
        self.skipFramesRadioButton = QtGui.QRadioButton(Dialog)
        self.skipFramesRadioButton.setGeometry(QtCore.QRect(30, 140, 351, 22))
        self.skipFramesRadioButton.setObjectName(_fromUtf8("skipFramesRadioButton"))
        self.binMeanRadioButton = QtGui.QRadioButton(Dialog)
        self.binMeanRadioButton.setGeometry(QtCore.QRect(30, 180, 371, 22))
        self.binMeanRadioButton.setObjectName(_fromUtf8("binMeanRadioButton"))
        self.binMaxRadioButton = QtGui.QRadioButton(Dialog)
        self.binMaxRadioButton.setGeometry(QtCore.QRect(30, 220, 351, 22))
        self.binMaxRadioButton.setObjectName(_fromUtf8("binMaxRadioButton"))
        self.binSizeLabel = QtGui.QLabel(Dialog)
        self.binSizeLabel.setEnabled(False)
        self.binSizeLabel.setGeometry(QtCore.QRect(30, 64, 51, 17))
        self.binSizeLabel.setObjectName(_fromUtf8("binSizeLabel"))
        self.binSizeLineEdit = QtGui.QLineEdit(Dialog)
        self.binSizeLineEdit.setEnabled(False)
        self.binSizeLineEdit.setGeometry(QtCore.QRect(90, 60, 113, 27))
        self.binSizeLineEdit.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.binSizeLineEdit.setText(_fromUtf8(""))
        self.binSizeLineEdit.setObjectName(_fromUtf8("binSizeLineEdit"))
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(30, 270, 131, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.numberOfFramesLabel = QtGui.QLabel(Dialog)
        self.numberOfFramesLabel.setGeometry(QtCore.QRect(170, 270, 91, 17))
        self.numberOfFramesLabel.setText(_fromUtf8(""))
        self.numberOfFramesLabel.setObjectName(_fromUtf8("numberOfFramesLabel"))
        self.label_4 = QtGui.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(30, 13, 151, 31))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.numberOfDataPointsLabel = QtGui.QLabel(Dialog)
        self.numberOfDataPointsLabel.setGeometry(QtCore.QRect(190, 20, 121, 17))
        self.numberOfDataPointsLabel.setText(_fromUtf8(""))
        self.numberOfDataPointsLabel.setObjectName(_fromUtf8("numberOfDataPointsLabel"))
        self.useDefaultPushButton = QtGui.QPushButton(Dialog)
        self.useDefaultPushButton.setGeometry(QtCore.QRect(280, 260, 97, 27))
        self.useDefaultPushButton.setObjectName(_fromUtf8("useDefaultPushButton"))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Dialog", None, QtGui.QApplication.UnicodeUTF8))
        self.okayPushButton.setText(QtGui.QApplication.translate("Dialog", "Okay", None, QtGui.QApplication.UnicodeUTF8))
        self.noBinRadioButton.setText(QtGui.QApplication.translate("Dialog", "No Bin - Display all Data Points", None, QtGui.QApplication.UnicodeUTF8))
        self.skipFramesRadioButton.setText(QtGui.QApplication.translate("Dialog", "Skip Frames - Display every Bin Size-th Data Point", None, QtGui.QApplication.UnicodeUTF8))
        self.binMeanRadioButton.setText(QtGui.QApplication.translate("Dialog", "Bin Frames - Display MEAN of Bin Size Data Points", None, QtGui.QApplication.UnicodeUTF8))
        self.binMaxRadioButton.setText(QtGui.QApplication.translate("Dialog", "Bin Frames - Display MAX of Bin Size Data Points", None, QtGui.QApplication.UnicodeUTF8))
        self.binSizeLabel.setText(QtGui.QApplication.translate("Dialog", "Bin Size", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Number Of frames:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Dialog", "Available Data Points:", None, QtGui.QApplication.UnicodeUTF8))
        self.useDefaultPushButton.setText(QtGui.QApplication.translate("Dialog", "Default", None, QtGui.QApplication.UnicodeUTF8))

