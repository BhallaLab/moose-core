# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'styleSelector.ui'
#
# Created: Wed Dec 28 16:50:58 2011
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
        Dialog.resize(661, 466)
        Dialog.setAccessibleName(_fromUtf8(""))
        self.filterLineEdit = QtGui.QLineEdit(Dialog)
        self.filterLineEdit.setGeometry(QtCore.QRect(100, 30, 161, 27))
        self.filterLineEdit.setInputMethodHints(QtCore.Qt.ImhNone)
        self.filterLineEdit.setText(_fromUtf8(""))
        self.filterLineEdit.setObjectName(_fromUtf8("filterLineEdit"))
        self.addToolButton = QtGui.QToolButton(Dialog)
        self.addToolButton.setGeometry(QtCore.QRect(200, 70, 24, 25))
        self.addToolButton.setObjectName(_fromUtf8("addToolButton"))
        self.minusToolButton = QtGui.QToolButton(Dialog)
        self.minusToolButton.setGeometry(QtCore.QRect(240, 70, 24, 25))
        self.minusToolButton.setObjectName(_fromUtf8("minusToolButton"))
        self.allToolButton = QtGui.QToolButton(Dialog)
        self.allToolButton.setGeometry(QtCore.QRect(280, 70, 24, 25))
        self.allToolButton.setObjectName(_fromUtf8("allToolButton"))
        self.styleComboBox = QtGui.QComboBox(Dialog)
        self.styleComboBox.setGeometry(QtCore.QRect(20, 70, 161, 27))
        self.styleComboBox.setObjectName(_fromUtf8("styleComboBox"))
        self.styleComboBox.addItem(_fromUtf8(""))
        self.styleComboBox.addItem(_fromUtf8(""))
        self.styleComboBox.addItem(_fromUtf8(""))
        self.styleComboBox.addItem(_fromUtf8(""))
        self.styleComboBox.addItem(_fromUtf8(""))
        self.label = QtGui.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(40, 34, 51, 17))
        self.label.setObjectName(_fromUtf8("label"))
        self.okayPushButton = QtGui.QPushButton(Dialog)
        self.okayPushButton.setGeometry(QtCore.QRect(530, 410, 97, 27))
        self.okayPushButton.setObjectName(_fromUtf8("okayPushButton"))
        self.useDefaultsPushButton = QtGui.QPushButton(Dialog)
        self.useDefaultsPushButton.setGeometry(QtCore.QRect(370, 410, 97, 27))
        self.useDefaultsPushButton.setObjectName(_fromUtf8("useDefaultsPushButton"))
        self.listWidget = QtGui.QListWidget(Dialog)
        self.listWidget.setGeometry(QtCore.QRect(20, 110, 281, 241))
        self.listWidget.setObjectName(_fromUtf8("listWidget"))
        self.offsetPositionLineEdit = QtGui.QLineEdit(Dialog)
        self.offsetPositionLineEdit.setGeometry(QtCore.QRect(150, 370, 151, 27))
        self.offsetPositionLineEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.offsetPositionLineEdit.setObjectName(_fromUtf8("offsetPositionLineEdit"))
        self.offsetAngleLineEdit = QtGui.QLineEdit(Dialog)
        self.offsetAngleLineEdit.setGeometry(QtCore.QRect(150, 410, 151, 27))
        self.offsetAngleLineEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.offsetAngleLineEdit.setObjectName(_fromUtf8("offsetAngleLineEdit"))
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(30, 370, 111, 17))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(30, 410, 101, 17))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.canvasContentsListWidget = QtGui.QListWidget(Dialog)
        self.canvasContentsListWidget.setGeometry(QtCore.QRect(320, 70, 321, 321))
        self.canvasContentsListWidget.setObjectName(_fromUtf8("canvasContentsListWidget"))
        self.label_4 = QtGui.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(320, 40, 121, 17))
        self.label_4.setObjectName(_fromUtf8("label_4"))

        self.retranslateUi(Dialog)
        self.styleComboBox.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Style Selector", None, QtGui.QApplication.UnicodeUTF8))
        self.addToolButton.setText(QtGui.QApplication.translate("Dialog", "+", None, QtGui.QApplication.UnicodeUTF8))
        self.minusToolButton.setText(QtGui.QApplication.translate("Dialog", "-", None, QtGui.QApplication.UnicodeUTF8))
        self.allToolButton.setText(QtGui.QApplication.translate("Dialog", "A", None, QtGui.QApplication.UnicodeUTF8))
        self.styleComboBox.setItemText(0, QtGui.QApplication.translate("Dialog", "Disks", None, QtGui.QApplication.UnicodeUTF8))
        self.styleComboBox.setItemText(1, QtGui.QApplication.translate("Dialog", "Ball & Sticks", None, QtGui.QApplication.UnicodeUTF8))
        self.styleComboBox.setItemText(2, QtGui.QApplication.translate("Dialog", "Cylinders", None, QtGui.QApplication.UnicodeUTF8))
        self.styleComboBox.setItemText(3, QtGui.QApplication.translate("Dialog", "Capsules", None, QtGui.QApplication.UnicodeUTF8))
        self.styleComboBox.setItemText(4, QtGui.QApplication.translate("Dialog", "Pyramids", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "Filter", None, QtGui.QApplication.UnicodeUTF8))
        self.okayPushButton.setText(QtGui.QApplication.translate("Dialog", "OK", None, QtGui.QApplication.UnicodeUTF8))
        self.useDefaultsPushButton.setText(QtGui.QApplication.translate("Dialog", "Use Default", None, QtGui.QApplication.UnicodeUTF8))
        self.offsetPositionLineEdit.setText(QtGui.QApplication.translate("Dialog", "0.0,0.0,0.0", None, QtGui.QApplication.UnicodeUTF8))
        self.offsetAngleLineEdit.setText(QtGui.QApplication.translate("Dialog", "0.0,0.0,0.0,0.0", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Offset Position", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Dialog", "Offset Angle", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Dialog", "Canvas Contents", None, QtGui.QApplication.UnicodeUTF8))

