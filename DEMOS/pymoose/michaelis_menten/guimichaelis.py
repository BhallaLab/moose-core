#!/usr/bin/env python
# -*- coding: utf-8 -*-
# guimichaelis.py --- 
# 
# Filename: guimichaelis.py
# Description: 
# Author: Gael Jalowicki
# Maintainer: 
# Created: Sat Jul 17 11:57:39 2010 (+0530)
# Version: 
# Last-Updated: Tue Jul 20 17:42:56 2010 (+0530)
#           By:
#     Update #:
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

from PyQt4.Qt import Qt
from PyQt4 import QtCore, QtGui
import PyQt4.Qwt5 as Qwt

from Michaelis import *
from Help import *

class MainWindow(QtGui.QDialog):
	def __init__(self,parent = None, fl = 0):
		QtGui.QDialog.__init__(self,parent)
		self.setWindowTitle("Michaelis Menten")
		self.setFixedSize(QtCore.QSize(460, 520))

		self.michaelis = Michaelis()
		self.bug = False

		self.buttonsFrame()
		self.modeFrame()
		self.chooseModeFrame()
		self.kineticsFrame()
		self.timeFrame()

		self.initActions()
		self.setDefaultMode()

		self.createPlotPanel()
		self.plotPanel.setWindowFlags(QtCore.Qt.Window)
		self.plotPanel.setWindowTitle(self.tr("Graphs"))
		self.plotPanel.setAttribute(QtCore.Qt.WA_DeleteOnClose)
		self.connect(self.plotPanel, QtCore.SIGNAL("destroyed()"), QtGui.qApp, QtCore.SLOT("closeAllWindows()"))


	def initActions(self):
		"""Initialize the actions.
		"""
		self.resetAction = QtGui.QAction(self.tr("Reset"), self)
		self.resetAction.setShortcut(self.tr("Ctrl+R"))
		self.resetAction.setStatusTip(self.tr("Reset the simulation state"))
		QtCore.QObject.connect(self.pushButton_reset, QtCore.SIGNAL("clicked()"), self.resetSlot)
       
		self.runAction = QtGui.QAction(self.tr("Run"), self)
		self.runAction.setShortcut(self.tr("Ctrl+X"))
		self.runAction.setStatusTip(self.tr("Run the simulation"))
		QtCore.QObject.connect(self.pushButton_run, QtCore.SIGNAL("clicked()"), self.runSlot)

		self.quitAction = QtGui.QAction(self.tr("Quit"), self)
		self.quitAction.setShortcut(self.tr("Ctrl+Q"))
		self.quitAction.setStatusTip(self.tr("Quit the Michaelis demo"))
		QtCore.QObject.connect(self.pushButton_quit, QtCore.SIGNAL("clicked()"), QtGui.qApp, QtCore.SLOT("closeAllWindows()"))

		self.helpAction = QtGui.QAction(self.tr("Help"), self)
		self.helpAction.setShortcut(self.tr("Ctrl+H"))
		self.helpAction.setStatusTip(self.tr("Open the help window"))
		QtCore.QObject.connect(self.pushButton_help, QtCore.SIGNAL("clicked()"), self.helpSlot)

		self.switchAction = QtGui.QAction(self.tr("Mode"), self)
		self.switchAction.setShortcut(self.tr("Ctrl+S"))
		self.switchAction.setStatusTip(self.tr("Switch the observation mode"))
		QtCore.QObject.connect(self.checkBox_kinetics, QtCore.SIGNAL("clicked()"), self.switchSlotMichaelis)
		QtCore.QObject.connect(self.checkBox_michaelis, QtCore.SIGNAL("clicked()"), self.switchSlotKinetics)

		self.formAction = QtGui.QAction(self.tr("Form"), self)
		self.formAction.setShortcut(self.tr("Ctrl+F"))
		self.formAction.setStatusTip(self.tr("Switch the enzyme form"))
		QtCore.QObject.connect(self.checkBox_expForm, QtCore.SIGNAL("clicked()"), self.switchSlotImplicit)
		QtCore.QObject.connect(self.checkBox_impForm, QtCore.SIGNAL("clicked()"), self.switchSlotExplicit)


	def helpSlot(self):
		"""Display a help window.
		"""
		self.helpFrame = QtGui.QFrame()

		self.helpTextEdit = QtGui.QTextEdit(self.helpFrame)

		htmlText = Help()
		self.helpTextEdit.setHtml(htmlText.text)
		self.helpTextEdit.setReadOnly(True)
		self.helpTextEdit.setFixedSize(QtCore.QSize(400, 800))

		self.helpFrame.setWindowFlags(QtCore.Qt.Window)
		self.helpFrame.setWindowTitle(self.tr("Help"))
#		self.helpFrame.setFixedSize(QtCore.QSize(100, 100))
		self.helpFrame.show()


	def runSlot(self):
		"""Reset the entire plots and simulation, then run and plot.
		"""
		# Resetting part.
		self.sub_prd_Plot.clear()
		self.enzPlot.clear()
		self.vPlot.clear()
		self.sub_prd_Plot.replot()
		self.enzPlot.replot()
		self.vPlot.replot()
		self.plotPanel.hide()
		self.resetSlot()
		# Running part.
		self.michaelis.RUNTIME = float(self.lineEdit_runtime.text())
		self.michaelis.doRun()
		subConcTableArray = numpy.array(self.michaelis._tabsub())
		prdConcTableArray = numpy.array(self.michaelis._tabprd())
		enzConcTableArray = numpy.array(self.michaelis._tabenzmol())
		self.michaelis.calculatingV()
		velocTableArray = numpy.array(self.michaelis._tabv())
		xData = numpy.linspace(0, self.michaelis.RUNTIME, len(subConcTableArray))
		print "length xData ", len(xData)
		self.plotPanel.show()
		self.addCurve(self.sub_prd_Plot, "substrate", QtCore.Qt.blue, xData, subConcTableArray, "product", QtCore.Qt.red, prdConcTableArray)
		if self.bug:
			self.michaelis.calculatingComplex()
			cplxConcTableArray = numpy.array(self.michaelis._tabcomplx2())
		else:
			cplxConcTableArray = numpy.array(self.michaelis._tabcomplx())
		self.addCurve(self.enzPlot, "complex", QtCore.Qt.blue, xData, cplxConcTableArray, "free\nenzyme", QtCore.Qt.red, enzConcTableArray)
		self.addCurve(self.vPlot, "velocity", QtCore.Qt.blue, xData, velocTableArray, "", "", "")


	def switchSlotExplicit(self):
		"""Switch the enzyme form.
		"""
		self.checkBox_expForm.setChecked(not self.checkBox_expForm.isChecked())

	def switchSlotImplicit(self):
		"""Switch the enzyme form.
		"""
		self.checkBox_impForm.setChecked(not self.checkBox_impForm.isChecked())		

	def getForm(self):
		"""Return the current enzyme mode.
		"""
		return self.checkBox_impForm.isChecked()

	def switchSlotMichaelis(self):
		"""Perform the mode switch if the michaelis checkBox is clicked.
		"""
		self.checkBox_michaelis.setChecked(not self.checkBox_michaelis.isChecked())
		self.switchLineEdit()

	def switchSlotKinetics(self):
		"""Perform the mode switch if the kinetics checkBox is clicked.
		"""
		self.checkBox_kinetics.setChecked(not self.checkBox_kinetics.isChecked())
		self.switchLineEdit()

	def switchLineEdit(self):
		"""Handle the mode switch.
		"""
		self.lineEdit_Km.setEnabled(self.checkBox_michaelis.isChecked())
		self.lineEdit_Vm.setEnabled(self.checkBox_michaelis.isChecked())
		self.lineEdit_ratio.setEnabled(self.checkBox_michaelis.isChecked())
		self.lineEdit_k1.setDisabled(self.checkBox_michaelis.isChecked())
		self.lineEdit_k2.setDisabled(self.checkBox_michaelis.isChecked())
		self.lineEdit_k3.setDisabled(self.checkBox_michaelis.isChecked())


	def resetSlot(self):
		"""Modify the model values through a dictionary.
		"""
		if (not self.getForm()) and self.michaelis.enz.mode:
			self.michaelis.paramDict["enzForm"] = self.getForm()
			self.bug = True
		else:
			self.michaelis.paramDict["enzForm"] = self.getForm()

		self.michaelis.paramDict["simDt"] = float(self.lineEdit_simdt.text())
		self.michaelis.paramDict["plotDt"] = float(self.lineEdit_plotdt.text())
		self.michaelis.paramDict["runtime"] = float(self.lineEdit_runtime.text())
		self.michaelis.paramDict["subConc"] = float(self.lineEdit_sub.text())
		self.michaelis.paramDict["prdConc"] = float(self.lineEdit_prd.text())
		self.michaelis.paramDict["enzConc"] = float(self.lineEdit_enz.text())
		self.michaelis.paramDict["volume"] = float(self.lineEdit_vol.text())
		self.michaelis.paramDict["Km"] = float(self.lineEdit_Km.text())
		self.michaelis.paramDict["Vm"] = float(self.lineEdit_Vm.text())
		self.michaelis.paramDict["k1"] = float(self.lineEdit_k1.text())
		self.michaelis.paramDict["k2"] = float(self.lineEdit_k2.text())
		self.michaelis.paramDict["k3"] = float(self.lineEdit_k3.text())
		self.michaelis.paramDict["ratio"] = float(self.lineEdit_ratio.text())
		if self.checkBox_michaelis.isChecked():
			self.michaelis.paramDict["k3"] = self.michaelis.paramDict["Vm"]/self.michaelis.paramDict["enzConc"]
			self.michaelis.paramDict["k2"] = self.michaelis.paramDict["k3"]*self.michaelis.paramDict["ratio"]
			self.michaelis.paramDict["k1"] = (self.michaelis.paramDict["k2"]+self.michaelis.paramDict["k3"])/self.michaelis.paramDict["Km"]
			self.lineEdit_k1.setText(str(self.michaelis.paramDict["k1"]))
			self.lineEdit_k2.setText(str(self.michaelis.paramDict["k2"]))
			self.lineEdit_k3.setText(str(self.michaelis.paramDict["k3"]))
		else:
			self.michaelis.paramDict["Km"] = (self.michaelis.paramDict["k2"]+self.michaelis.paramDict["k3"])/self.michaelis.paramDict["k1"]
			self.michaelis.paramDict["Vm"] = self.michaelis.paramDict["k3"]*self.michaelis.paramDict["enzConc"]
			self.michaelis.paramDict["ratio"] = self.michaelis.paramDict["k2"]/self.michaelis.paramDict["k3"]
			self.lineEdit_Km.setText(str(self.michaelis.paramDict["Km"]))
			self.lineEdit_Vm.setText(str(self.michaelis.paramDict["Vm"]))
			self.lineEdit_ratio.setText(str(self.michaelis.paramDict["ratio"]))
		self.michaelis.doResetForMichaelis()

	def setDefaultMode(self):
		"""Set a starting mode.
		"""
		self.lineEdit_Km.setDisabled(True)
		self.lineEdit_Vm.setEnabled(False)
		self.lineEdit_ratio.setDisabled(True)
		self.checkBox_michaelis.setChecked(False)
		self.checkBox_kinetics.setChecked(True)
		self.checkBox_expForm.setChecked(True)
		self.checkBox_impForm.setChecked(False)

	def timeFrame(self):
		"""A frame containing the time and enzyme form settings.
		"""
		self.frame_time = QtGui.QFrame(self)
		self.frame_time.setGeometry(QtCore.QRect(0,235,460,130))
		self.frame_time.setFrameShape(QtGui.QFrame.StyledPanel)
		self.frame_time.setFrameShadow(QtGui.QFrame.Raised)

		# groupBox time settings
		self.groupBox_time = QtGui.QGroupBox("groupBox_time", self.frame_time)
		self.groupBox_time.setTitle(self.__tr("Clocks Settings"))
		self.groupBox_time.setGeometry(QtCore.QRect(10,10,219,110))
		groupBox_timeLayout = QtGui.QGridLayout(self.groupBox_time.layout())
		groupBox_timeLayout.setAlignment(Qt.AlignTop)

		self.textLabel_simdt = QtGui.QLabel(self.groupBox_time)
		self.textLabel_simdt.setText(self.__tr("SimDt (s)"))
		groupBox_timeLayout.addWidget(self.textLabel_simdt,0,0)

		self.lineEdit_simdt = QtGui.QLineEdit(str(self.michaelis._simDt()), self.groupBox_time)
		groupBox_timeLayout.addWidget(self.lineEdit_simdt,0,1)

		self.textLabel_plotdt = QtGui.QLabel(self.groupBox_time)
		self.textLabel_plotdt.setText(self.__tr("PlotDt (s)"))
		groupBox_timeLayout.addWidget(self.textLabel_plotdt,1,0)

		self.lineEdit_plotdt = QtGui.QLineEdit(str(self.michaelis._plotDt()), self.groupBox_time)
		groupBox_timeLayout.addWidget(self.lineEdit_plotdt,1,1)

		self.textLabel_runtime = QtGui.QLabel(self.groupBox_time)
		self.textLabel_runtime.setText(self.__tr("Runtime (s)"))
		groupBox_timeLayout.addWidget(self.textLabel_runtime,2,0)

		self.lineEdit_runtime = QtGui.QLineEdit(str(self.michaelis._runtime()), self.groupBox_time)
		groupBox_timeLayout.addWidget(self.lineEdit_runtime,2,1)

		self.groupBox_time.setLayout(groupBox_timeLayout)

		# groupBox form settings
		self.groupBox_form = QtGui.QGroupBox("groupBox_form", self.frame_time)
		self.groupBox_form.setTitle(self.__tr("Enzyme Form"))
		self.groupBox_form.setGeometry(QtCore.QRect(240,10,219,110))
		groupBox_formLayout = QtGui.QGridLayout(self.groupBox_form.layout())
		groupBox_formLayout.setAlignment(Qt.AlignTop)

		self.checkBox_expForm = QtGui.QCheckBox(self.frame_time)
		self.checkBox_expForm.setText(self.__tr("Explicit"))

		self.checkBox_impForm = QtGui.QCheckBox(self.frame_time)
		self.checkBox_impForm.setText(self.__tr("Implicit"))

		groupBox_formLayout.addWidget(self.checkBox_expForm, 0, 0)
		groupBox_formLayout.addWidget(self.checkBox_impForm, 1, 0)
		self.groupBox_form.setLayout(groupBox_formLayout)


	def kineticsFrame(self):
		"""A frame containing concentrations and volume settings.
		"""
		self.frame_kin = QtGui.QFrame(self)
		self.frame_kin.setGeometry(QtCore.QRect(0,365,460,160))
		self.frame_kin.setFrameShape(QtGui.QFrame.StyledPanel)
		self.frame_kin.setFrameShadow(QtGui.QFrame.Raised)

		self.groupBox_kin = QtGui.QGroupBox(self.frame_kin)
		self.groupBox_kin.setTitle(self.__tr("Kinetics Settings"))
		self.groupBox_kin.setGeometry(QtCore.QRect(10,10,430,140))

		groupBox_kinLayout = QtGui.QGridLayout(self.groupBox_kin)
		groupBox_kinLayout.setAlignment(Qt.AlignTop)

		self.textLabel_vol = QtGui.QLabel("Volume", self.groupBox_kin)
		self.textLabel_vol.setText(self.__tr("Volume (Liters)"))
		groupBox_kinLayout.addWidget(self.textLabel_vol,0,0)

		self.lineEdit_vol = QtGui.QLineEdit(str(self.michaelis._volume()), self.groupBox_kin)
		groupBox_kinLayout.addWidget(self.lineEdit_vol,0,1)

		self.textLabel_sub = QtGui.QLabel("textLabel_mode_12", self.groupBox_kin)
		self.textLabel_sub.setText(self.__tr("Substrate Concentration (uM)"))
		groupBox_kinLayout.addWidget(self.textLabel_sub,1,0)

		self.lineEdit_sub = QtGui.QLineEdit(str(self.michaelis._subConc()), self.groupBox_kin)
		groupBox_kinLayout.addWidget(self.lineEdit_sub,1,1)

		self.textLabel_enz = QtGui.QLabel("textLabel_mode_13", self.groupBox_kin)
		self.textLabel_enz.setText(self.__tr("Enzyme Concentration (uM)"))
		groupBox_kinLayout.addWidget(self.textLabel_enz,2,0)

		self.lineEdit_enz = QtGui.QLineEdit(str(self.michaelis._enzConc()), self.groupBox_kin)
		groupBox_kinLayout.addWidget(self.lineEdit_enz,2,1)

		self.textLabel_prd = QtGui.QLabel("textLabel_mode_14", self.groupBox_kin)
		self.textLabel_prd.setText(self.__tr("Product Concentration (uM)"))
		groupBox_kinLayout.addWidget(self.textLabel_prd,3,0)

		self.lineEdit_prd = QtGui.QLineEdit(str(self.michaelis._prdConc()), self.groupBox_kin)
		groupBox_kinLayout.addWidget(self.lineEdit_prd,3,1)

		self.groupBox_kin.setLayout(groupBox_kinLayout)


	def modeFrame(self):
		"""Here we handle the switch events.
		"""
		self.frame_mode = QtGui.QFrame(self)
		self.frame_mode.setGeometry(QtCore.QRect(0,95,460,140))
		self.frame_mode.setFrameShape(QtGui.QFrame.StyledPanel)
		self.frame_mode.setFrameShadow(QtGui.QFrame.Raised)

		self.groupBox_mode = QtGui.QGroupBox("groupBox_mode", self.frame_mode)
		self.groupBox_mode.setTitle(self.__tr("Constants Settings"))
		self.groupBox_mode.setGeometry(QtCore.QRect(10,10,430,120))

		groupBox_modeLayout = QtGui.QGridLayout(self.groupBox_mode)
		groupBox_modeLayout.setAlignment(Qt.AlignTop)

		self.textLabel_Km = QtGui.QLabel("textLabel_mode_1", self.groupBox_mode)
		self.textLabel_Km.setText(self.__tr("Km (uM)"))
		groupBox_modeLayout.addWidget(self.textLabel_Km,0,0)

		self.lineEdit_Km = QtGui.QLineEdit(str(self.michaelis._Km()), self.groupBox_mode)
		groupBox_modeLayout.addWidget(self.lineEdit_Km,0,1)

		self.textLabel_Vm = QtGui.QLabel("textLabel_mode_2", self.groupBox_mode)
		self.textLabel_Vm.setText(self.__tr("Vm (uM/s)"))
		groupBox_modeLayout.addWidget(self.textLabel_Vm,1,0)

		self.lineEdit_Vm = QtGui.QLineEdit(str(self.michaelis._Vm()), self.groupBox_mode)
		groupBox_modeLayout.addWidget(self.lineEdit_Vm,1,1)

		self.textLabel_ratio = QtGui.QLabel("", self.groupBox_mode)
		self.textLabel_ratio.setText(self.__tr("Ratio k2/k3"))
		groupBox_modeLayout.addWidget(self.textLabel_ratio,2,0)

		self.lineEdit_ratio = QtGui.QLineEdit(str(self.michaelis._ratio()), self.groupBox_mode)
		groupBox_modeLayout.addWidget(self.lineEdit_ratio,2,1)

		self.textLabel_k1 = QtGui.QLabel("textLabel_mode_4", self.groupBox_mode)
		self.textLabel_k1.setText(self.__tr("k1"))
		groupBox_modeLayout.addWidget(self.textLabel_k1,0,2)

		self.lineEdit_k1 = QtGui.QLineEdit(str(self.michaelis._k1()), self.groupBox_mode)
		groupBox_modeLayout.addWidget(self.lineEdit_k1,0,3)

		self.textLabel_k2 = QtGui.QLabel("textLabel_mode_5", self.groupBox_mode)
		self.textLabel_k2.setText(self.__tr("k2"))
		groupBox_modeLayout.addWidget(self.textLabel_k2,1,2)

		self.lineEdit_k2 = QtGui.QLineEdit(str(self.michaelis._k2()), self.groupBox_mode)
		groupBox_modeLayout.addWidget(self.lineEdit_k2,1,3)

		self.textLabel_k3 = QtGui.QLabel("textLabel_mode_6", self.groupBox_mode)
		self.textLabel_k3.setText(self.__tr("k3"))
		groupBox_modeLayout.addWidget(self.textLabel_k3,2,2)

		self.lineEdit_k3 = QtGui.QLineEdit(str(self.michaelis._k3()), self.groupBox_mode)
		groupBox_modeLayout.addWidget(self.lineEdit_k3,2,3)
		self.groupBox_mode.setLayout(groupBox_modeLayout)


	def chooseModeFrame(self):
		"""A frame with 2 checkBoxes, one per mode.
		"""
		self.chooseframe = QtGui.QFrame(self)
		self.chooseframe.setGeometry(QtCore.QRect(0,50,460,45))
		self.chooseframe.setFrameShape(QtGui.QFrame.StyledPanel)
		self.chooseframe.setFrameShadow(QtGui.QFrame.Raised)

		chooseframeLayout = QtGui.QHBoxLayout(self.chooseframe)

		self.checkBox_michaelis = QtGui.QCheckBox(self.chooseframe)
		chooseframeLayout.addWidget(self.checkBox_michaelis)

		self.checkBox_kinetics = QtGui.QCheckBox(self.chooseframe)
		chooseframeLayout.addWidget(self.checkBox_kinetics)

		self.checkBox_kinetics.setText(self.__tr("Kinetics mode"))
		self.checkBox_michaelis.setText(self.__tr("Michaelis mode"))



	def buttonsFrame(self):
		"""A frame with 3 Buttons handling quit, reset and run.
		"""
		self.framebutton = QtGui.QFrame(self)
		self.framebutton.setGeometry(QtCore.QRect(0,0,460,50))
		self.framebutton.setFrameShape(QtGui.QFrame.StyledPanel)
		self.framebutton.setFrameShadow(QtGui.QFrame.Raised)

		framebuttonLayout = QtGui.QHBoxLayout(self.framebutton)

		self.pushButton_quit = QtGui.QPushButton(self.framebutton)
		self.pushButton_quit.setEnabled(1)
		self.pushButton_quit.setText(self.__tr("Quit"))
		framebuttonLayout.addWidget(self.pushButton_quit)

		self.pushButton_help = QtGui.QPushButton(self.framebutton)
		self.pushButton_help.setEnabled(1)
		self.pushButton_help.setText(self.__tr("Help"))
		framebuttonLayout.addWidget(self.pushButton_help)

		self.pushButton_reset = QtGui.QPushButton(self.framebutton)
		self.pushButton_reset.setEnabled(1)
		self.pushButton_reset.setText(self.__tr("Reset"))
		framebuttonLayout.addWidget(self.pushButton_reset)

		self.pushButton_run = QtGui.QPushButton(self.framebutton)
		self.pushButton_run.setEnabled(1)
		self.pushButton_run.setText(self.__tr("Reset and Run"))
		framebuttonLayout.addWidget(self.pushButton_run)

		self.setTabOrder(self.pushButton_quit, self.pushButton_help)
		self.setTabOrder(self.pushButton_help, self.pushButton_reset)
		self.setTabOrder(self.pushButton_reset, self.pushButton_run)


	def __tr(self,s,c = None):
		return app.translate("Form1",s,c)

	def createPlotFrame(self, parent, plotName, xLabel, yLabel):
		"""
		This function creates a curve with curveName of colour "color", x axis
		with xLabel and yAxis with yLabel.
		"""
		if hasattr(self, plotName):
			return getattr(self, plotName)
		frame = QtGui.QFrame(parent)
		frame.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
		plot = Qwt.QwtPlot(frame)
		setattr(self, plotName, plot)
		plot.insertLegend(Qwt.QwtLegend(), Qwt.QwtPlot.RightLegend)
		plot.setAxisTitle(Qwt.QwtPlot.xBottom, xLabel)
		plot.setAxisTitle(Qwt.QwtPlot.yLeft, yLabel)
		plot.replot()
		layout = QtGui.QVBoxLayout(frame)
		layout.addWidget(plot)
		frame.setLayout(layout)
		return frame


	def createPlotPanel(self):
		"""A frame where are put the plots.
		"""
		self.plotPanel = QtGui.QFrame()
		sub_prd_PlotFrame = self.createPlotFrame(
			self.plotPanel,
			"sub_prd_Plot",
			"<font size=2>Time</font>",
			"<font size=2>Concentration</font>")
		self.sub_prd_Plot.setTitle("<font size=2>Substrate and Product Concentrations</font>")
		enzymePlotFrame = self.createPlotFrame(
			self.plotPanel,
			"enzPlot",
			"<font size=2>Time</font>",
			"<font size=2>Concentration</font>")
		self.enzPlot.setTitle("<font size=2>Enzyme Concentration</font>")
		velocityPlotFrame = self.createPlotFrame(
			self.plotPanel,
			"vPlot",
			"<font size=2>Time</font>",
			"<font size=2>Velocity</font>")
		self.vPlot.setTitle("<font size=2>Reaction Velocity</font>")

		layout = QtGui.QGridLayout(self.plotPanel)
		layout.addWidget(sub_prd_PlotFrame, 0, 0)
		layout.addWidget(enzymePlotFrame, 0, 1)
		layout.addWidget(velocityPlotFrame, 1, 0)
		layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
		self.plotPanel.setLayout(layout)


	def addCurve(self, plot, curveName1, color1, xData, yData1, curveName2, color2, yData2):
		"""Add curve with curveName to the plot using color.
		"""
		self.makeCurve(plot, curveName1, color1, xData, yData1)
		if curveName2 <> "":
			self.makeCurve(plot, curveName2, color2, xData, yData2)


	def makeCurve(self, plot, curveName, color, xData, yData):
		"""Create curves and attach them to the QwtPlot object.
		"""
		curve = Qwt.QwtPlotCurve(self.tr(curveName))
		curve.setPen(color)
		curve.setData(xData, yData)
		curve.attach(plot)
		plot.replot()
		if not hasattr(plot, "zoomer"):
			zoomer = Qwt.QwtPlotZoomer(Qwt.QwtPlot.xBottom,
				Qwt.QwtPlot.yLeft,
				Qwt.QwtPicker.DragSelection,
				Qwt.QwtPicker.AlwaysOn,
				plot.canvas())
			zoomer.setRubberBandPen(QtGui.QPen(QtCore.Qt.white))
			zoomer.setTrackerPen(QtGui.QPen(QtCore.Qt.red))
			plot.zoomer = zoomer
			plot.zoomer.setZoomBase()
		plot.replot()


import sys
if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)
	QtCore.QObject.connect(app, QtCore.SIGNAL('lastWindowClosed()'), app, QtCore.SLOT('quit()'))
	fileName = None
	fileType = None
	if len(sys.argv) == 3:
		fileName = sys.argv[1]
		fileType = sys.argv[2]
	elif len(sys.argv) == 2:
		errorDialog = QtGui.QErrorMessage()
		errorDialog.setWindowFlags(Qt.WindowStaysOnTopHint)
		errorDialog.showMessage('<p>If specifying a file to load, you must specify a model type as the final argument to this program</p>')
		app.exec_()
	mainwin = MainWindow()

	mainwin.show()
	app.exec_()


#
# guimichaelis.py ends here
