#!/usr/bin/env python
#
# This is the GUI and controls for the squid demo
import sys
from squidModel import *
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4 import Qwt5 as Qwt
import PyQt4.Qwt5.anynumpy as numpy



STATEPLOT_COLORMAP = { "V-h": QtCore.Qt.red, "h-V": QtCore.Qt.darkRed, "m-n": QtCore.Qt.green, "n-m": QtCore.Qt.darkGreen, "n-V":QtCore.Qt.blue, "V-n":QtCore.Qt.darkBlue, "h-n": QtCore.Qt.cyan, "n-h": QtCore.Qt.darkCyan, "m-h": QtCore.Qt.magenta, "h-m": QtCore.Qt.darkMagenta, "V-m": QtCore.Qt.lightGray, "m-V": QtCore.Qt.gray, "V-V": QtCore.Qt.black, "h-h":QtCore.Qt.black, "m-m": QtCore.Qt.black, "n-n": QtCore.Qt.black }

COLORLIST = [ QtCore.Qt.red, 
              QtCore.Qt.blue, 
              QtCore.Qt.darkYellow, 
              QtCore.Qt.green, 
              QtCore.Qt.magenta, 
              QtCore.Qt.darkCyan,
              QtCore.Qt.black, 
              QtCore.Qt.cyan, 
              QtCore.Qt.darkRed, 
              QtCore.Qt.darkGreen, 
              QtCore.Qt.yellow, 
              QtCore.Qt.darkMagenta,  
              QtCore.Qt.gray,   
              QtCore.Qt.darkBlue, 
              QtCore.Qt.lightGray ]



class QtSquid(QtGui.QMainWindow):
    """The squid demo using PyQt4 and PyQwt5"""
    def __init__(self, *args):
        QtGui.QMainWindow.__init__(self,*args)
        self.setWindowTitle("SquidAxon")
        self.overlay = False
        self.runCount = -1
        self.connect(self, QtCore.SIGNAL("destroyed()"), QtGui.qApp, QtCore.SLOT("closeAllWindows()"))
        self.statePlotParams = {"V": [], "m": [], "h": [], "n": []}
        self.statePlotWindow = self.createStatePlotWindow()
        self.statePlotWindow.setWindowFlags(QtCore.Qt.Window)
        self.statePlotWindow.setWindowTitle(self.tr("State plot"))
        self.statePlotWindow.hide()
        self.initActions()
        self.initToolBar()
        self.createInputPanel()
        self.createPlotPanel()
        self.setCentralWidget(self.inputPanel)
        self.plotPanel.setWindowFlags(QtCore.Qt.Window)
        self.plotPanel.setWindowTitle(self.tr("Graphs"))
        self.plotPanel.setAttribute(QtCore.Qt.WA_DeleteOnClose) # This will make this emit destroyed() signal on close
        self.connect(self.plotPanel, QtCore.SIGNAL("destroyed()"), QtGui.qApp, QtCore.SLOT("closeAllWindows()"))
        self.plotPanel.show()
        self.squidModel = SquidModel("/squidModel")
        self.runNo = 0
    # !QtSquid.__init__

    def initMenubar(self):
        self.fileMenu = self.menuBar().addMenu(self.tr("&File"))
        self.fileMenu.addAction(self.quitAction)
    # !initMenubar

    def initToolBar(self):
        self.simToolBar = self.addToolBar(self.tr("Simulation Control"))
        self.simToolBar.addAction(self.quitAction)
        self.simToolBar.addAction(self.resetAction)
        self.simToolBar.addAction(self.runAction)
#         self.simToolBar.addAction(self.plotAction)
        self.simToolBar.addAction(self.statePlotAction)
        self.simToolBar.addAction(self.overlayAction)
    # !initToolBar

    def initActions(self):
        """Initialize actions"""
        self.resetAction = QtGui.QAction(self.tr("Reset"), self)
        self.resetAction.setShortcut(self.tr("Ctrl+R"))
        self.resetAction.setStatusTip(self.tr("Reset the simulation state"))
        self.connect(self.resetAction, QtCore.SIGNAL("triggered()"), self.resetSlot)
        
        self.runAction = QtGui.QAction(self.tr("Run"), self)
        self.runAction.setShortcut(self.tr("Ctrl+X"))
        self.runAction.setStatusTip(self.tr("Run the simulation"))
        self.connect(self.runAction, QtCore.SIGNAL("triggered()"), self.runSlot)

        self.quitAction = QtGui.QAction(self.tr("Quit"), self)
        self.quitAction.setShortcut(self.tr("Ctrl+Q"))
        self.quitAction.setStatusTip(self.tr("Quit the Squid Axon demo"))
        self.connect(self.quitAction, QtCore.SIGNAL("triggered()"), QtGui.qApp, QtCore.SLOT("closeAllWindows()"))

        self.plotAction = QtGui.QAction(self.tr("PlotWindow"), self)
        self.plotAction.setStatusTip(self.tr("Show/ Hide plot window"))
        self.connect(self.plotAction, QtCore.SIGNAL("triggered()"), self.plotWindowSlot)

        self.statePlotAction = QtGui.QAction(self.tr("StatePlot"), self)
        self.statePlotAction.setStatusTip(self.tr("Show state plot in a pop up window"))
        self.connect(self.statePlotAction, QtCore.SIGNAL("triggered()"), self.statePlotSlot)
        
        self.overlayAction = QtGui.QAction(self.tr("Overlay"), self)
        self.overlayAction.setStatusTip(self.tr("Overlay the plots"))
        self.overlayAction.setCheckable(True)
        self.connect(self.overlayAction, QtCore.SIGNAL("triggered()"), self.overlaySlot)

        # !initActions


    def createSimCtrl(self):
        # Create the inputs: runtime, simulation stepsize, plot stepsize
        self.simInputBox = QtGui.QGroupBox(self)
        self.runTimeLabel = QtGui.QLabel("Run time (ms)", self)
        self.simTimeStepLabel = QtGui.QLabel("Simulation time step (ms)", self)
        self.plotTimeStepLabel = QtGui.QLabel("Plotting interval (ms)", self)
        self.runTimeEdit = QtGui.QLineEdit("50.0", self)
        self.simTimeStepEdit = QtGui.QLineEdit("0.01", self)
        self.plotTimeStepEdit = QtGui.QLineEdit("0.1", self)
        layout = QtGui.QGridLayout(self.simInputBox)
        layout.addWidget(self.runTimeLabel, 0,0)
        layout.addWidget(self.runTimeEdit, 0, 1)
        layout.addWidget(self.simTimeStepLabel, 1, 0)
        layout.addWidget(self.simTimeStepEdit, 1, 1)
        layout.addWidget(self.plotTimeStepLabel, 2, 0)
        layout.addWidget(self.plotTimeStepEdit, 2, 1)
        self.simInputBox.setLayout(layout)
    # !createSimInput

    def createChannelCtrl(self):
        self.channelCtrlBox = QtGui.QGroupBox("Control ion channels",self)
        self.naChannelBlock = QtGui.QCheckBox("Na channel blocked", self)
        self.kChannelBlock  = QtGui.QCheckBox("K channel blocked", self)
        channelCtrlLayout = QtGui.QVBoxLayout(self.channelCtrlBox)
        channelCtrlLayout.addWidget(self.naChannelBlock)
        channelCtrlLayout.addWidget(self.kChannelBlock)
        self.channelCtrlBox.setLayout(channelCtrlLayout)
    # !QtSquid.createChannelCtrl

    def createConcCtrl(self):
        self.concCtrlBox = QtGui.QGroupBox("External ion concentration", self)
        self.temperatureLabel = QtGui.QLabel("Temperature (K)", self)
        self.temperatureEdit = QtGui.QLineEdit("279.45", self)
        self.kConcLabel = QtGui.QLabel("[K] mM", self)
        self.kConcEdit = QtGui.QLineEdit("10.0",self)
        self.naConcLabel = QtGui.QLabel("[Na] mM", self)
        self.naConcEdit = QtGui.QLineEdit("460.0",self)
        concCtrlLayout = QtGui.QGridLayout(self.concCtrlBox)
        concCtrlLayout.addWidget(self.kConcLabel, 0, 0)
        concCtrlLayout.addWidget(self.kConcEdit, 0, 1)
        concCtrlLayout.addWidget(self.naConcLabel, 1, 0)
        concCtrlLayout.addWidget(self.naConcEdit, 1, 1)
        concCtrlLayout.addWidget(self.temperatureLabel, 2, 0)
        concCtrlLayout.addWidget(self.temperatureEdit, 2, 1)

        self.concCtrlBox.setLayout(concCtrlLayout)
    # !QtSquid.createConcCtrl

    def createClampTypeCtrl(self):
        # Choice for Voltage or Current clamp
        self.clampTypeCombo = QtGui.QComboBox(self)
        self.clampTypeCombo.addItem("Current Clamp")
        self.clampTypeCombo.addItem("Voltage Clamp")
        self.clampTypeCombo.setEditable(False)
        self.connect(self.clampTypeCombo, QtCore.SIGNAL("activated(int)"), self.toggleClampSlot)
    # !QtSquid.createClampTypeCtrl

    def createIClampCtrl(self):
        iClampPanel = QtGui.QGroupBox("Current-Clamp Settings", self)
        self.iClampCtrlBox = iClampPanel
        self.baseCurrentLabel = QtGui.QLabel("Base Current Level (uA)",iClampPanel)
        self.baseCurrentEdit = QtGui.QLineEdit("0.0",iClampPanel)
        self.firstPulseLabel = QtGui.QLabel("First Pulse Current (uA)", iClampPanel)
        self.firstPulseEdit = QtGui.QLineEdit("0.1", iClampPanel)
        self.firstDelayLabel = QtGui.QLabel("First Onset Delay (ms)", iClampPanel)
        self.firstDelayEdit = QtGui.QLineEdit("5.0",iClampPanel)
        self.firstPulseWidthLabel = QtGui.QLabel("First Pulse Width (ms)", iClampPanel)
        self.firstPulseWidthEdit = QtGui.QLineEdit("40.0", iClampPanel)
        self.secondPulseLabel = QtGui.QLabel("Second Pulse Current (uA)", iClampPanel)
        self.secondPulseEdit = QtGui.QLineEdit("0.0", iClampPanel)
        self.secondDelayLabel = QtGui.QLabel("Second Onset Delay (ms)", iClampPanel)
        self.secondDelayEdit = QtGui.QLineEdit("0.0",iClampPanel)
        self.secondPulseWidthLabel = QtGui.QLabel("Second Pulse Width (ms)", iClampPanel)
        self.secondPulseWidthEdit = QtGui.QLineEdit("0.0", iClampPanel)
        self.pulseMode = QtGui.QComboBox(iClampPanel)
        self.pulseMode.addItem("Single Pulse")
        self.pulseMode.addItem("Pulse Train")
        layout = QtGui.QGridLayout(iClampPanel)
        layout.addWidget(self.baseCurrentLabel, 0, 0)
        layout.addWidget(self.baseCurrentEdit, 0, 1)
        layout.addWidget(self.firstPulseLabel, 1, 0)
        layout.addWidget(self.firstPulseEdit, 1, 1)
        layout.addWidget(self.firstDelayLabel, 2, 0)
        layout.addWidget(self.firstDelayEdit, 2, 1)
        layout.addWidget(self.firstPulseWidthLabel, 3, 0)
        layout.addWidget(self.firstPulseWidthEdit, 3, 1)
        layout.addWidget(self.secondPulseLabel, 4, 0)
        layout.addWidget(self.secondPulseEdit, 4, 1)
        layout.addWidget(self.secondDelayLabel, 5, 0)
        layout.addWidget(self.secondDelayEdit, 5, 1)
        layout.addWidget(self.secondPulseWidthLabel, 6, 0)
        layout.addWidget(self.secondPulseWidthEdit, 6, 1)
        layout.addWidget(self.pulseMode, 7, 0, 1, 2)
        layout.setSizeConstraint(QtGui.QLayout.SetFixedSize)
        iClampPanel.setLayout(layout)
    #! QtSquid.createIClampCtrl

    def createVClampCtrl(self):
        vClampPanel = QtGui.QGroupBox("Voltage-Clamp Settings", self)
        self.vClampCtrlBox = vClampPanel
        self.holdingVLabel = QtGui.QLabel("Holding Voltage (mV)", vClampPanel)
        self.holdingVEdit = QtGui.QLineEdit("0.0", vClampPanel)
        self.holdingTimeLabel = QtGui.QLabel("Holding Time (ms)", vClampPanel)
        self.holdingTimeEdit = QtGui.QLineEdit("10.0", vClampPanel)
        self.prePulseVLabel = QtGui.QLabel("Pre-pulse Voltage (mV)", vClampPanel)
        self.prePulseVEdit = QtGui.QLineEdit("0.0", vClampPanel)
        self.prePulseTimeLabel = QtGui.QLabel("Pre-pulse Time (ms)", vClampPanel)
        self.prePulseTimeEdit = QtGui.QLineEdit("0.0", vClampPanel)
        self.clampVLabel = QtGui.QLabel("Clamp Voltage (mV)", vClampPanel)
        self.clampVEdit = QtGui.QLineEdit("50.0", vClampPanel)
        self.clampTimeLabel = QtGui.QLabel("Clamp Time (ms)", vClampPanel)
        self.clampTimeEdit = QtGui.QLineEdit("20.0", vClampPanel)
        self.pidGainValueSlider = QtGui.QSlider(QtCore.Qt.Horizontal, vClampPanel)
        self.pidGainValueSlider.setRange(0, 9)
        self.pidGainValueSlider.setTickPosition(self.pidGainValueSlider.TicksBelow)
        self.pidGainValueSlider.setTickInterval(1)
        self.pidGainValueSlider.setValue(3)
        self.pidGainExpSlider = QtGui.QSlider(QtCore.Qt.Horizontal, vClampPanel)
        self.pidGainExpSlider.setRange(-9, 0)
        self.pidGainExpSlider.setTickPosition(self.pidGainExpSlider.TicksBelow)
        self.pidGainExpSlider.setTickInterval(1)
        self.pidGainExpSlider.setValue(-4)
        self.pidGainLabel = QtGui.QLabel("PID Gain")
        self.pidGainValueLabel = QtGui.QLabel("(mantissa [0..9])")
        self.pidGainExpLabel = QtGui.QLabel("(exponent [-9..0])")
        layout = QtGui.QGridLayout(vClampPanel)
        layout.addWidget(self.holdingVLabel, 0, 0)
        layout.addWidget(self.holdingVEdit, 0, 1)
        layout.addWidget(self.holdingTimeLabel, 1, 0)
        layout.addWidget(self.holdingTimeEdit, 1, 1)
        layout.addWidget(self.prePulseVLabel, 2, 0)
        layout.addWidget(self.prePulseVEdit, 2, 1)
        layout.addWidget(self.prePulseTimeLabel,3,0)
        layout.addWidget(self.prePulseTimeEdit, 3, 1)
        layout.addWidget(self.clampVLabel, 4, 0)
        layout.addWidget(self.clampVEdit, 4, 1)
        layout.addWidget(self.clampTimeLabel, 5, 0)
        layout.addWidget(self.clampTimeEdit, 5, 1)
        layout.addWidget(self.pidGainLabel, 6, 0)
        layout.addWidget(self.pidGainValueSlider, 7, 0)
        layout.addWidget(self.pidGainExpSlider, 7, 1)
        layout.addWidget(self.pidGainValueLabel, 8, 0)
        layout.addWidget(self.pidGainExpLabel, 8, 1)
        
        vClampPanel.setLayout(layout)
    #! QtSquid.createVClampCtrl

    
    def createInputPanel(self):
        """Create the input panel"""
        self.inputPanel = QtGui.QFrame(self)
        self.inputPanel.setFrameStyle(QtGui.QFrame.StyledPanel)
        self.createSimCtrl()
        self.createChannelCtrl()
        self.createConcCtrl()
        self.createClampTypeCtrl()
        self.createIClampCtrl()
        self.createVClampCtrl()
        self.vClampCtrlBox.hide()
        layout = QtGui.QVBoxLayout(self.inputPanel)
        layout.addWidget(self.simInputBox)
        layout.addWidget(self.channelCtrlBox)
        layout.addWidget(self.concCtrlBox)
        layout.addWidget(self.clampTypeCombo)
        layout.addWidget(self.iClampCtrlBox)
        layout.addWidget(self.vClampCtrlBox)
        layout.setSizeConstraint(QtGui.QLayout.SetFixedSize)
        self.inputPanel.setLayout(layout)

    def _createPlotFrame(self, parent, plotName, xLabel, yLabel, curveNameColorPairs):
        """
        This function creates a curve with curveName of colour "color", x axis
        with xLabel and yAxis with yLabel
        """
        if hasattr(self, plotName):
            return getattr(self, plotName)
        frame = QtGui.QFrame(parent)
        frame.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        plot = Qwt.QwtPlot( frame)
        setattr(self, plotName, plot)
        plot.insertLegend(Qwt.QwtLegend(), Qwt.QwtPlot.RightLegend)
        plot.setAxisTitle(Qwt.QwtPlot.xBottom, xLabel)
        plot.setAxisTitle(Qwt.QwtPlot.yLeft, yLabel)
        x = numpy.arange(0.0, 6.28, 0.1)
        y = [numpy.sin(x),numpy.cos(x)]
        plot.replot()
        layout = QtGui.QVBoxLayout(frame)
        layout.addWidget(plot)
        frame.setLayout(layout)
        return frame
    
    def createPlotPanel(self):
        """
        This function creates the plot panel and the curves.
        The follwoing attributes are created:
        plotPanel - QWidget
        vmPlot - QwtPlot
        vmPlot.Vm - QwtPlotCurve
        vmPlot.Command - QwtPlotCurve
        injectionPlot - QwtPlot
        injectionPlot.IClamp - QwtPlotCurve
        injectionPlot.VClamp - QwtPlotCurve
        conductancePlot - QwtPlot
        conductancePlot.Na- QwtPlotCurve
        conductancePlot.K - QwtPlotCurve
        chanCurrentPlot - QwtPlot
        chanCurrentPlot.Na - QwtPlotCurve
        chanCurrentPlot.K - QwtPlotCurve
        """
        self.plotPanel = QtGui.QFrame()
        vmPlotFrame = self._createPlotFrame(
            self.plotPanel, 
            "vmPlot",
            "t (ms) ->",
            "Vm (mV) ->",
            [("Vm",QtCore.Qt.red), ("Command", QtCore.Qt.blue)])
        self.vmPlot.setTitle(self.tr("Membrane potential"))
        injectionPlotFrame = self._createPlotFrame(
            self.plotPanel,
            "injectionPlot",
            "t (ms) ->",
            "current (uA)",
            [("IClamp", QtCore.Qt.red), ("VClamp", QtCore.Qt.blue)])
        self.injectionPlot.setTitle(self.tr("Current injection"))
        conductancePlotFrame = self._createPlotFrame(
            self.plotPanel,
            "conductancePlot",
            "t (ms) ->",
            "conductance (mS) ->",
            [("Na", QtCore.Qt.red), ("K", QtCore.Qt.blue)])
        self.conductancePlot.setTitle(self.tr("Channel conductance"))
        chanCurrentPlotFrame = self._createPlotFrame(
            self.plotPanel,
            "chanCurrentPlot",
            "t (ms) ->",
            "channel current (uA) ->",
            [("Na", QtCore.Qt.red), ("K", QtCore.Qt.blue)])
        self.chanCurrentPlot.setTitle(self.tr("Channel current"))
        layout = QtGui.QGridLayout(self.plotPanel)
        layout.addWidget(vmPlotFrame, 0, 0)
        layout.addWidget(injectionPlotFrame, 1, 0)
        layout.addWidget(conductancePlotFrame, 0, 1)
        layout.addWidget(chanCurrentPlotFrame, 1, 1)
        layout.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.plotPanel.setLayout(layout)
        #!createPlotPanel
        
    def createStatePlotWindow(self):
        self.statePlotX = "V"
        self.statePlotY = "n"
        statePlotWin = QtGui.QWidget(self)
        statePlotFrame = self._createPlotFrame(
            statePlotWin,
            "statePlot",
            "V ->",
            "n ->",
            [("state", QtCore.Qt.black)])
        self.statePlot.setTitle("State Plot")
        activationPlotFrame = self._createPlotFrame(
            statePlotWin,
            "activationPlot",
            "t (ms) ->",
            "",
            [("m", QtCore.Qt.black), ("n", QtCore.Qt.blue), ("h", QtCore.Qt.red)])
        self.activationPlot.setTitle("H-H Activation parameters")
        inputFrame = self._createStatePlotInput(statePlotWin)
        layout = QtGui.QVBoxLayout(statePlotWin)
        layout.addWidget(statePlotFrame)
        layout.addWidget(inputFrame)
        layout.addWidget(activationPlotFrame)
        statePlotWin.setLayout(layout)
        return statePlotWin

    def _createStatePlotInput(self, parent):
        frame = QtGui.QFrame(parent)
        xCombo = QtGui.QComboBox(frame)
        xCombo.addItems(["V","m","n","h"])
        xCombo.setEditable(False)
        xCombo.setCurrentIndex(0)
        xCombo.connect(xCombo, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.statePlotXSlot) 
        yCombo = QtGui.QComboBox(frame)
        yCombo.addItems(["V","m","n","h"])
        yCombo.setCurrentIndex(2)
        yCombo.setEditable(False)
        yCombo.connect(yCombo, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.statePlotYSlot) 
        clearButton = QtGui.QPushButton("Clear", frame)
        self.connect(clearButton, QtCore.SIGNAL("clicked(bool)"), self.clearStatePlotSlot)
        layout = QtGui.QHBoxLayout(frame)
        layout.addWidget(xCombo)
        layout.addWidget(yCombo)
        layout.addWidget(clearButton)
        frame.setLayout(layout)
        return frame

    # Think of how reset should behave? The necessity of resetting
    # every time you change something is UNCOOL! I would expect the
    # software to take care of everything - in run method. But for the
    # time being I shall go for backward compatibility.
    def resetSlot(self):
        """Reset the simulation"""
        if not self.overlay:
            print "Overlay is off. Clearing all plots."
            self.runCount = -1
            self.vmPlot.clear()
            self.injectionPlot.clear()
            self.conductancePlot.clear()
            self.chanCurrentPlot.clear()
            self.statePlot.clear()
            self.activationPlot.clear()
            self.vmPlot.replot()
            self.injectionPlot.replot()
            self.conductancePlot.replot()
            self.chanCurrentPlot.replot()
            self.statePlot.replot()
            self.activationPlot.replot()

        # The stuff below could as well go to the run method.
        simdt = float(self.simTimeStepEdit.text())
        plotdt = float(self.plotTimeStepEdit.text())
        self.clampType = self.clampTypeCombo.currentIndex()
#         self.squidModel.blockNaChannel(self.naChannelBlock.isChecked())
#         self.squidModel.blockKChannel(self.kChannelBlock.isChecked())
        paramDict = {}
        # We do the unit conversion before sending it to the model
        # The model takes everything as SI units
        if self.clampType == 0: # Current Clamp
            paramDict["baseLevel"] = 1e-6 * float(self.baseCurrentEdit.text())
            paramDict["firstDelay"] = 1e-3 * float(self.firstDelayEdit.text())
            paramDict["firstWidth"] = 1e-3 * float(self.firstPulseWidthEdit.text())
            paramDict["firstLevel"] = 1e-6 * float(self.firstPulseEdit.text())
            paramDict["secondLevel"] = 1e-6 * float(self.secondPulseEdit.text())
            paramDict["secondDelay"] = 1e-3 * float(self.secondDelayEdit.text())
            paramDict["secondWidth"] = 1e-3 * float(self.secondPulseWidthEdit.text())
            paramDict["simDt"] = 1e-3 * float(self.simTimeStepEdit.text())
            paramDict["plotDt"] = 1e-3 * float(self.plotTimeStepEdit.text())
            # index = 0 -> single pulse -> singlePulse = True
            # index = 1 -> train of pulses -> singlePulse = False
            paramDict["singlePulse"] = (self.pulseMode.currentIndex() == 0)
            paramDict["blockNa"] = self.naChannelBlock.isChecked()
            paramDict["blockK"] = self.kChannelBlock.isChecked()
            paramDict["temperature"] = float(self.temperatureEdit.text())
            paramDict["naConc"] = float(self.naConcEdit.text())
            paramDict["kConc"] = float(self.kConcEdit.text())
            self.squidModel.doResetForIClamp(paramDict)
        else:
            print "Voltage clamp"
            paramDict["baseLevel"] = 1e-3 * float(self.holdingVEdit.text())
            paramDict["firstDelay"] = 1e-3 * float(self.holdingTimeEdit.text())
            paramDict["firstWidth"] = 1e-3 * float(self.prePulseTimeEdit.text())
            paramDict["firstLevel"] = 1e-3 * float(self.prePulseVEdit.text())
            paramDict["secondLevel"] = 1e-3 * float(self.clampVEdit.text())
            paramDict["secondDelay"] = 1e-3 * paramDict["firstWidth"]
            paramDict["secondWidth"] = 1e-3 * float(self.clampTimeEdit.text())
            paramDict["simDt"] = 1e-3 * float(self.simTimeStepEdit.text())
            paramDict["plotDt"] = 1e-3 * float(self.plotTimeStepEdit.text())
            paramDict["singlePulse"] = (self.pulseMode.currentIndex() == 0)
            paramDict["blockNa"] = self.naChannelBlock.isChecked()
            paramDict["blockK"] = self.kChannelBlock.isChecked()
            paramDict["temperature"] = float(self.temperatureEdit.text())
            paramDict["naConc"] = float(self.naConcEdit.text())
            paramDict["kConc"] = float(self.kConcEdit.text())
            self.squidModel.doResetForVClamp(paramDict)
            self.squidModel._PID.gain = float(self.pidGainValueSlider.value()) * pow(10.0, float(self.pidGainExpSlider.value()))
            


    def addCurve(self, plot, curveName, color, xData, yData):
        """Add curve with curveName to the plot using color."""
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
            zoomer.setTrackerPen(QtGui.QPen(QtCore.Qt.cyan))
            plot.zoomer = zoomer
            plot.zoomer.setZoomBase()
        plot.replot()

    def runSlot(self):
        """Run the simulation"""
        runtime = 1e-3*float(self.runTimeEdit.text())
        self.squidModel.doRun(runtime)
        # Convert the tables into numpy arrays. 
        # Also convert the SI units to physiological units
        vmTable = numpy.array(self.squidModel.vmTable()) * 1e3 
        injectionTable = numpy.array(self.squidModel.iInjectTable()) * 1e6 
        iNaTable = numpy.array(self.squidModel.iNaTable()) * 1e6 
        iKTable = numpy.array(self.squidModel.iKTable()) * 1e6 
        gNaTable = numpy.array(self.squidModel.gNaTable()) * 1e3 
        gKTable = numpy.array(self.squidModel.gKTable()) * 1e3 
        nParamTable = numpy.array(self.squidModel.nParamTable())
        mParamTable = numpy.array(self.squidModel.mParamTable())
        hParamTable = numpy.array(self.squidModel.hParamTable())
        xData = numpy.linspace(0,runtime,len(vmTable)) * 1e3 
        
        if self.overlay:
            self.runCount = self.runCount + 1
        else:
            self.runCount = 0

        color0 = self.runCount % len(COLORLIST)
        color1 = ( 2 * self.runCount ) % len(COLORLIST)
        color2 = ( 2 * self.runCount + 1) % len(COLORLIST)

        self.addCurve(self.vmPlot, "Vm#"+str(self.runCount), COLORLIST[color0], xData, vmTable)
        self.addCurve(self.injectionPlot, "I #"+str(self.runCount), COLORLIST[color0], xData, injectionTable)
        self.addCurve(self.chanCurrentPlot, "Na#"+str(self.runCount), COLORLIST[color1], xData, iNaTable)
        self.addCurve(self.chanCurrentPlot, "K #"+str(self.runCount), COLORLIST[color2], xData, iKTable)
        self.addCurve(self.conductancePlot, "Na#"+str(self.runCount), COLORLIST[color1], xData, gNaTable)
        self.addCurve(self.conductancePlot, "K #"+str(self.runCount), COLORLIST[color2], xData, gKTable)
        self.statePlotParams = {"V":vmTable, "n":nParamTable, "m":mParamTable, "h":hParamTable}
        key = self.statePlotX + "-" + self.statePlotY
        self.addCurve(self.statePlot, key, STATEPLOT_COLORMAP[key],
                      self.statePlotParams[self.statePlotX], self.statePlotParams[self.statePlotY])
        self.addCurve(self.activationPlot, "m",
                      QtCore.Qt.black, xData, mParamTable)
        self.addCurve(self.activationPlot, "h",
                      QtCore.Qt.red, xData, hParamTable)
        self.addCurve(self.activationPlot, "n",
                      QtCore.Qt.blue, xData, nParamTable)
        self.squidModel.dumpPlotData()

    def statePlotSlot(self):
        """Show state plot"""
        if self.statePlotWindow.isVisible():
            self.statePlotWindow.hide()
        else:
            self.statePlotWindow.show()

    def statePlotXSlot(self, newValue):
        self.statePlotX = str(newValue)
        key = self.statePlotX + "-" + self.statePlotY
        self.statePlot.setAxisTitle(Qwt.QwtPlot.xBottom, self.statePlotX)
        self.statePlot.setAxisTitle(Qwt.QwtPlot.yLeft, self.statePlotY)

        self.addCurve(self.statePlot, key, STATEPLOT_COLORMAP[key],
                      self.statePlotParams[self.statePlotX], self.statePlotParams[self.statePlotY])
        self.statePlot.replot()
        
    def statePlotYSlot(self, newValue):
        self.statePlotY = str(newValue)
        key = self.statePlotX + "-" + self.statePlotY
        self.statePlot.setAxisTitle(Qwt.QwtPlot.xBottom, self.statePlotX)
        self.statePlot.setAxisTitle(Qwt.QwtPlot.yLeft, self.statePlotY)
        self.addCurve(self.statePlot, key, STATEPLOT_COLORMAP[key],
                      self.statePlotParams[self.statePlotX], self.statePlotParams[self.statePlotY])
        self.statePlot.replot()

    def clearStatePlotSlot(self):
        if hasattr(self, "statePlot"):
            self.statePlot.clear()
            self.statePlot.replot()

    def overlaySlot(self):
        """Toggle overlay state"""
        if self.overlayAction.isChecked():
            self.overlay = True
        else:
            self.overlay = False

    def toggleClampSlot(self,  index):
        """Switch between voltage and current clamp"""
        if index == 0:
            self.vClampCtrlBox.hide()
            self.iClampCtrlBox.show()
        else:
            self.vClampCtrlBox.show()
            self.iClampCtrlBox.hide()

    def plotWindowSlot(self):
        """Toggle visibility of plotPanel"""
        if self.plotPanel.isVisible():
            self.plotPanel.hide()
        else:
            self.plotPanel.show()



    
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    QtGui.qApp = app
    app.connect(app, QtCore.SIGNAL('lastWindowClosed()'), app, QtCore.SLOT('quit()'))
    qtSquid = QtSquid()
    qtSquid.show()
    sys.exit(app.exec_())
