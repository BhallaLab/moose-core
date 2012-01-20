# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'vizSelector.ui'
#
# Created: Fri Jan 13 17:00:50 2012
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
        Dialog.resize(660, 462)
        Dialog.setAccessibleName(_fromUtf8(""))
        self.filterLineEdit = QtGui.QLineEdit(Dialog)
        self.filterLineEdit.setGeometry(QtCore.QRect(20, 50, 171, 27))
        self.filterLineEdit.setInputMethodHints(QtCore.Qt.ImhNone)
        self.filterLineEdit.setText(_fromUtf8(""))
        self.filterLineEdit.setObjectName(_fromUtf8("filterLineEdit"))
        self.addToolButton = QtGui.QToolButton(Dialog)
        self.addToolButton.setGeometry(QtCore.QRect(200, 50, 24, 25))
        self.addToolButton.setObjectName(_fromUtf8("addToolButton"))
        self.minusToolButton = QtGui.QToolButton(Dialog)
        self.minusToolButton.setGeometry(QtCore.QRect(240, 50, 24, 25))
        self.minusToolButton.setObjectName(_fromUtf8("minusToolButton"))
        self.allToolButton = QtGui.QToolButton(Dialog)
        self.allToolButton.setGeometry(QtCore.QRect(280, 50, 24, 25))
        self.allToolButton.setObjectName(_fromUtf8("allToolButton"))
        self.label = QtGui.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 20, 51, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.okayPushButton = QtGui.QPushButton(Dialog)
        self.okayPushButton.setGeometry(QtCore.QRect(530, 410, 97, 27))
        self.okayPushButton.setObjectName(_fromUtf8("okayPushButton"))
        self.useDefaultsPushButton = QtGui.QPushButton(Dialog)
        self.useDefaultsPushButton.setGeometry(QtCore.QRect(390, 410, 97, 27))
        self.useDefaultsPushButton.setObjectName(_fromUtf8("useDefaultsPushButton"))
        self.listWidget = QtGui.QListWidget(Dialog)
        self.listWidget.setGeometry(QtCore.QRect(20, 90, 281, 241))
        self.listWidget.setObjectName(_fromUtf8("listWidget"))
        self.vizContentsListWidget = QtGui.QListWidget(Dialog)
        self.vizContentsListWidget.setGeometry(QtCore.QRect(320, 50, 321, 341))
        self.vizContentsListWidget.setObjectName(_fromUtf8("vizContentsListWidget"))
        self.label_4 = QtGui.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(320, 20, 211, 17))
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.colorMapLabelComboBox = QtGui.QComboBox(Dialog)
        self.colorMapLabelComboBox.setGeometry(QtCore.QRect(20, 370, 85, 27))
        self.colorMapLabelComboBox.setObjectName(_fromUtf8("colorMapLabelComboBox"))
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(20, 340, 71, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.newColorMapPushButton = QtGui.QPushButton(Dialog)
        self.newColorMapPushButton.setGeometry(QtCore.QRect(120, 370, 81, 27))
        self.newColorMapPushButton.setObjectName(_fromUtf8("newColorMapPushButton"))
        self.minDisplayPropertyLabel = QtGui.QLabel(Dialog)
        self.minDisplayPropertyLabel.setGeometry(QtCore.QRect(30, 410, 51, 31))
        self.minDisplayPropertyLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.minDisplayPropertyLabel.setWordWrap(False)
        self.minDisplayPropertyLabel.setObjectName(_fromUtf8("minDisplayPropertyLabel"))
        self.editColorMapPushButton = QtGui.QPushButton(Dialog)
        self.editColorMapPushButton.setGeometry(QtCore.QRect(220, 370, 81, 27))
        self.editColorMapPushButton.setObjectName(_fromUtf8("editColorMapPushButton"))
        self.colorMapToolButton = QtGui.QToolButton(Dialog)
        self.colorMapToolButton.setGeometry(QtCore.QRect(90, 410, 141, 25))
        self.colorMapToolButton.setIconSize(QtCore.QSize(128, 24))
        self.colorMapToolButton.setAutoRaise(True)
        self.colorMapToolButton.setObjectName(_fromUtf8("colorMapToolButton"))
        self.maxDisplayPropertyLabel = QtGui.QLabel(Dialog)
        self.maxDisplayPropertyLabel.setGeometry(QtCore.QRect(240, 410, 51, 31))
        self.maxDisplayPropertyLabel.setObjectName(_fromUtf8("maxDisplayPropertyLabel"))

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Visualization Options", None, QtGui.QApplication.UnicodeUTF8))
        self.addToolButton.setText(QtGui.QApplication.translate("Dialog", "+", None, QtGui.QApplication.UnicodeUTF8))
        self.minusToolButton.setText(QtGui.QApplication.translate("Dialog", "-", None, QtGui.QApplication.UnicodeUTF8))
        self.allToolButton.setText(QtGui.QApplication.translate("Dialog", "A", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "Filter", None, QtGui.QApplication.UnicodeUTF8))
        self.okayPushButton.setText(QtGui.QApplication.translate("Dialog", "OK", None, QtGui.QApplication.UnicodeUTF8))
        self.useDefaultsPushButton.setText(QtGui.QApplication.translate("Dialog", "Use Default", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Dialog", "Visualization Contents", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Color Map", None, QtGui.QApplication.UnicodeUTF8))
        self.newColorMapPushButton.setText(QtGui.QApplication.translate("Dialog", "New", None, QtGui.QApplication.UnicodeUTF8))
        self.minDisplayPropertyLabel.setText(QtGui.QApplication.translate("Dialog", "test", None, QtGui.QApplication.UnicodeUTF8))
        self.editColorMapPushButton.setText(QtGui.QApplication.translate("Dialog", "Edit", None, QtGui.QApplication.UnicodeUTF8))
        self.colorMapToolButton.setText(QtGui.QApplication.translate("Dialog", "...", None, QtGui.QApplication.UnicodeUTF8))
        self.maxDisplayPropertyLabel.setText(QtGui.QApplication.translate("Dialog", "test", None, QtGui.QApplication.UnicodeUTF8))

