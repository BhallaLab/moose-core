# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'colorMap.ui'
#
# Created: Fri Jan 13 17:07:27 2012
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
        Dialog.resize(262, 300)
        self.verticalLayout = QtGui.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label = QtGui.QLabel(Dialog)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_2 = QtGui.QLabel(Dialog)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_3 = QtGui.QLabel(Dialog)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_4 = QtGui.QLabel(Dialog)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 3, 0, 1, 1)
        self.labelLineEdit = QtGui.QLineEdit(Dialog)
        self.labelLineEdit.setObjectName(_fromUtf8("labelLineEdit"))
        self.gridLayout.addWidget(self.labelLineEdit, 0, 1, 1, 1)
        self.minValLineEdit = QtGui.QLineEdit(Dialog)
        self.minValLineEdit.setObjectName(_fromUtf8("minValLineEdit"))
        self.gridLayout.addWidget(self.minValLineEdit, 1, 1, 1, 1)
        self.maxValLineEdit = QtGui.QLineEdit(Dialog)
        self.maxValLineEdit.setObjectName(_fromUtf8("maxValLineEdit"))
        self.gridLayout.addWidget(self.maxValLineEdit, 2, 1, 1, 1)
        self.colorMapComboBox = QtGui.QComboBox(Dialog)
        self.colorMapComboBox.setObjectName(_fromUtf8("colorMapComboBox"))
        self.gridLayout.addWidget(self.colorMapComboBox, 3, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.colorMapToolButton = QtGui.QToolButton(Dialog)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.colorMapToolButton.sizePolicy().hasHeightForWidth())
        self.colorMapToolButton.setSizePolicy(sizePolicy)
        self.colorMapToolButton.setIconSize(QtCore.QSize(128, 24))
        self.colorMapToolButton.setAutoRaise(True)
        self.colorMapToolButton.setObjectName(_fromUtf8("colorMapToolButton"))
        self.horizontalLayout_2.addWidget(self.colorMapToolButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.okayPushButton = QtGui.QPushButton(Dialog)
        self.okayPushButton.setObjectName(_fromUtf8("okayPushButton"))
        self.verticalLayout.addWidget(self.okayPushButton)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "ColorMap", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Dialog", "Label", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Dialog", "Min Value", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Dialog", "Max Value", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Dialog", "ColorMap", None, QtGui.QApplication.UnicodeUTF8))
        self.colorMapToolButton.setText(QtGui.QApplication.translate("Dialog", "...", None, QtGui.QApplication.UnicodeUTF8))
        self.okayPushButton.setText(QtGui.QApplication.translate("Dialog", "OK", None, QtGui.QApplication.UnicodeUTF8))

