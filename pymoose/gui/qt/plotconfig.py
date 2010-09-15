# plotconfig.py --- 
# 
# Filename: plotconfig.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Fri Jul  9 00:21:51 2010 (+0530)
# Version: 
# Last-Updated: Wed Sep 15 20:23:14 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 337
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

from PyQt4.Qt import Qt
from PyQt4 import QtGui, QtCore
from PyQt4 import Qwt5 as Qwt

class PlotConfig(QtGui.QDialog):
    """Widget to visually configure plots."""

    curveStyleNames = [
        'NoCurve',
        'Lines',
        'Sticks',
        'Steps',
        'Dots'
        ]

    curveAttributeNames = [
        'None',
        'Fitted',
        'Inverted'
        ]
    curveAttributeMap = {
        'None': 0,
        'Fitted': Qwt.QwtPlotCurve.Fitted,
        'Inverted': Qwt.QwtPlotCurve.Inverted
        }
    curveStyleMap = {
        'NoCurve': Qwt.QwtPlotCurve.NoCurve,
        'Lines': Qwt.QwtPlotCurve.Lines,
        'Sticks': Qwt.QwtPlotCurve.Sticks,
        'Steps': Qwt.QwtPlotCurve.Steps,
        'Dots': Qwt.QwtPlotCurve.Dots
        }
        
    penStyleMap = {
        'NoPen': Qt.NoPen,
        'SolidLine': Qt.SolidLine,
        'DashLine': Qt.DashLine,
        'DotLine': Qt.DotLine,
        'DashDotLine': Qt.DashDotLine,
        'DashDotDotLine': Qt.DashDotDotLine
        }
    penStyleNames = [
        'NoPen',
        'SolidLine',
        'DashLine',
        'DotLine',
        'DashDotLine',
        'DashDotDotLine'
        ]

    symbolStyleNames = [
        'NoSymbol',
        'Ellipse',
        'Rect',
        'Diamond',
        'Triangle',
        'DTriangle',
        'UTriangle',
        'LTriangle',
        'RTriangle',
        'Cross',
        'XCross',
        'HLine',
        'VLine',
        'Star1',
        'Star2',
        'Hexagon',
        'StyleCnt'
        ]
    symbolStyleMap = {
        'NoSymbol': Qwt.QwtSymbol.NoSymbol,
        'Ellipse': Qwt.QwtSymbol.Ellipse,
        'Rect': Qwt.QwtSymbol.Rect,
        'Diamond': Qwt.QwtSymbol.Diamond,
        'Triangle': Qwt.QwtSymbol.Triangle,
        'DTriangle': Qwt.QwtSymbol.DTriangle,
        'UTriangle': Qwt.QwtSymbol.UTriangle,
        'LTriangle': Qwt.QwtSymbol.LTriangle,
        'RTriangle': Qwt.QwtSymbol.RTriangle,
        'Cross': Qwt.QwtSymbol.Cross,
        'XCross': Qwt.QwtSymbol.XCross,
        'HLine': Qwt.QwtSymbol.HLine,
        'VLine': Qwt.QwtSymbol.VLine,
        'Star1': Qwt.QwtSymbol.Star1,
        'Star2': Qwt.QwtSymbol.Star2,
        'Hexagon': Qwt.QwtSymbol.Hexagon,
        'StyleCnt': Qwt.QwtSymbol.StyleCnt
        }
    def __init__(self, *args):
        QtGui.QDialog.__init__(self, *args)
        self.currentLineColor = QtGui.QColor(Qt.black)
        self.currentSymbolPenColor = QtGui.QColor(Qt.black)
        self.currentSymbolFillColor = QtGui.QColor(Qt.black)
        
        layout = QtGui.QGridLayout()
        
        row = 0
        self.styleLabel = QtGui.QLabel(self.tr('Curve Style'), self)
        layout.addWidget(self.styleLabel, row, 0)

        self.styleCombo = QtGui.QComboBox(self)
        self.styleCombo.addItems(PlotConfig.curveStyleNames)
        self.styleCombo.setCurrentIndex(PlotConfig.curveStyleNames.index('Lines'))
        layout.addWidget(self.styleCombo, row, 1)

        row += 1
        self.attributeLabel = QtGui.QLabel(self.tr('Curve Attribute'), self)
        layout.addWidget(self.attributeLabel, row, 0)
        

        self.attributeCombo = QtGui.QComboBox(self)
        self.attributeCombo.addItems(PlotConfig.curveAttributeNames)
        self.attributeCombo.setCurrentIndex(0)
        layout.addWidget(self.attributeCombo, row, 1)

        row += 1
        self.penLabel = QtGui.QLabel(self.tr('Curve Pen'), self)
        self.penLabel.setFrameStyle(QtGui.QFrame.Box)
        layout.addWidget(self.penLabel, row, 0, 1, 2)
        
        row += 1
        self.lineColorLabel = QtGui.QLabel(self.tr('Color'), self)
        layout.addWidget(self.lineColorLabel, row, 0)

        self.lineColorButton = QtGui.QPushButton(self.currentLineColor.name(), self)
        self.lineColorButton.setPalette(QtGui.QPalette(self.currentLineColor))
        self.connect(self.lineColorButton, QtCore.SIGNAL('clicked()'), self.setLineColor)
        self.lineColorButton.setAutoFillBackground(True)
        self.lineColorButton.setObjectName('lineColorButton')
        layout.addWidget(self.lineColorButton, row, 1)
        
        row += 1
        self.lineWidthLabel = QtGui.QLabel(self.tr('Width'), self)
        layout.addWidget(self.lineWidthLabel, row, 0)
        
        self.lineWidthText = QtGui.QLineEdit('1', self)        
        layout.addWidget(self.lineWidthText, row, 1)

        row += 1
        self.lineStyleLabel = QtGui.QLabel(self.tr('Style'), self)
        layout.addWidget(self.lineStyleLabel, row, 0)

        self.lineStyleCombo = QtGui.QComboBox(self)
        self.lineStyleCombo.addItems(PlotConfig.penStyleNames)
        self.lineStyleCombo.setCurrentIndex(PlotConfig.penStyleNames.index('SolidLine'))
        layout.addWidget(self.lineStyleCombo, row, 1)
        
        # Options for plot symbol
        row += 1
        self.symbolLabel = QtGui.QLabel(self.tr('Plot Symbol'), self)
        self.symbolLabel.setFrameStyle(QtGui.QFrame.Box)
        layout.addWidget(self.symbolLabel, row, 0, 1, 2)
        
        row += 1
        self.symbolStyleLabel = QtGui.QLabel(self.tr('Style'), self)
        layout.addWidget(self.symbolStyleLabel, row, 0)

        self.symbolStyleCombo = QtGui.QComboBox(self)
        self.symbolStyleCombo.addItems(PlotConfig.symbolStyleNames)
        self.symbolStyleCombo.setCurrentIndex(PlotConfig.symbolStyleNames.index('NoSymbol'))
        layout.addWidget(self.symbolStyleCombo, row, 1)

        row += 1
        self.symbolPenColorlabel = QtGui.QLabel(self.tr('Pen Color'), self)
        layout.addWidget(self.symbolPenColorlabel, row, 0)
        
        self.symbolPenColorButton = QtGui.QPushButton(self.currentSymbolPenColor.name(), self)
        self.connect(self.symbolPenColorButton, QtCore.SIGNAL('clicked()'), self.setSymbolPenColor)
        self.symbolPenColorButton.setPalette(QtGui.QPalette(self.currentSymbolPenColor))
        self.symbolPenColorButton.setObjectName('symbolPenColorButton')
        self.symbolPenColorButton.setAutoFillBackground(True)
        layout.addWidget(self.symbolPenColorButton, row, 1)

        row += 1 
        self.symbolLineWidthLabel = QtGui.QLabel(self.tr('Width'), self)
        layout.addWidget(self.symbolLineWidthLabel, row, 0)

        self.symbolLineWidthText = QtGui.QLineEdit('1', self)        
        layout.addWidget(self.symbolLineWidthText, row, 1)

        row += 1
        self.symbolFillColorLabel = QtGui.QLabel(self.tr('Fill Color'), self)
        layout.addWidget(self.symbolFillColorLabel, row, 0)

        self.symbolFillColorButton = QtGui.QPushButton(self.currentSymbolFillColor.name(), self)
        self.connect(self.symbolFillColorButton, QtCore.SIGNAL('clicked()'), self.setSymbolFillColor)
        self.symbolFillColorButton.setPalette(QtGui.QPalette(self.currentSymbolFillColor))
        self.symbolFillColorButton.setAutoFillBackground(True)
        self.symbolFillColorButton.setObjectName('symbolFillColorButton')
        layout.addWidget(self.symbolFillColorButton, row, 1)

        row += 1
        self.symbolWidthLabel = QtGui.QLabel(self.tr('Width'), self)
        layout.addWidget(self.symbolWidthLabel, row, 0)

        self.symbolWidthText = QtGui.QLineEdit('3', self)        
        layout.addWidget(self.symbolWidthText, row, 1)

        row += 1
        self.symbolHeightLabel = QtGui.QLabel(self.tr('Height'), self)
        layout.addWidget(self.symbolHeightLabel, row, 0)

        self.symbolHeightText = QtGui.QLineEdit('3', self)        
        layout.addWidget(self.symbolHeightText, row, 1)

        row += 1
        self.cancelButton = QtGui.QPushButton(self.tr('Cancel'), self)
        self.connect(self.cancelButton, QtCore.SIGNAL('clicked()'), self.reject)
        layout.addWidget(self.cancelButton, row, 0)

        self.okButton = QtGui.QPushButton(self.tr('OK'), self)
        self.connect(self.okButton, QtCore.SIGNAL('clicked()'), self.accept)
        layout.addWidget(self.okButton, row, 1)

        self.okButton.setDefault(True)
        
        self.setLayout(layout)

    def setLineColor(self):
        currentColor = self.lineColorButton.palette().color(QtGui.QPalette.Background)
        color = QtGui.QColorDialog.getColor(currentColor, self)
        if color.isValid():
            style = QtCore.QString('QPushButton#lineColorButton {background-color: %s}' % color.name())
            self.lineColorButton.setStyleSheet(style)
            self.lineColorButton.setText(color.name())
            self.currentLineColor = color
        
    def setSymbolPenColor(self):
        currentColor = self.symbolPenColorButton.palette().color(QtGui.QPalette.Background)
        color = QtGui.QColorDialog.getColor(currentColor, self)
        if color.isValid():
            style = QtCore.QString('QPushButton#symbolPenColorButton {background-color: %s}' % color.name())
            self.symbolPenColorButton.setStyleSheet(style)
            self.symbolPenColorButton.setText(color.name())
            self.currentSymbolPenColor = color

    def setSymbolFillColor(self):
        currentColor = self.symbolFillColorButton.palette().color(QtGui.QPalette.Background)
        color = QtGui.QColorDialog.getColor(currentColor, self)
        if color.isValid():
            style = QtCore.QString('QPushButton#symbolFillColorButton {background-color: %s}' % color.name())
            self.symbolFillColorButton.setStyleSheet(style)
            self.symbolFillColorButton.setText(color.name())
            self.currentSymbolFillColor = color

    def getPen(self):
        """Return a QPen with the line properties."""
        pen = QtGui.QPen()
        pen.setColor(self.currentLineColor)
        pen.setStyle(self.penStyleMap[str(self.lineStyleCombo.currentText())])
        pen.setWidth(float(self.lineWidthText.text()))
        return pen

    def getSymbol(self):
        """return a QwtSymbol object with the selected settings."""
        style = PlotConfig.symbolStyleMap[str(self.symbolStyleCombo.currentText())]
        brush = QtGui.QBrush(self.currentSymbolFillColor)
        pen = QtGui.QPen(self.currentSymbolPenColor)
        pen.setWidth(float(self.symbolLineWidthText.text()))
        width = int(self.symbolWidthText.text())
        height = int(self.symbolHeightText.text())
        symbol = Qwt.QwtSymbol(style, brush, pen, QtCore.QSize(width, height))
        return symbol

    def getStyle(self):
        return PlotConfig.curveStyleMap[str(self.styleCombo.currentText())]

    def getAttribute(self):
        return PlotConfig.curveAttributeMap[str(self.attributeCombo.currentText())]

if __name__ == '__main__':
    app = QtGui.QApplication([])
    widget = PlotConfig()
    widget.show()
    app.exec_()



# 
# plotconfig.py ends here
