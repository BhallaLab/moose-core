import sys
import os
import shutil
import h5py
import numpy
import csv

from canvas import *
from moogliLayout import *
from defaults import *

from layout.styleSelector import Ui_Dialog as styleSelectDia
from layout.vizSelector import Ui_Dialog as paraSelectDia
from layout.colorMap import Ui_Dialog as colorMapDia
from layout.binProperty import Ui_Dialog as binPropertyDia
from layout.graphPlotter import Ui_MainWindow as gPlotWin

from PyQt4 import Qt,QtGui,QtCore

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

class DesignerMainWindow(QtGui.QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(DesignerMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.connectActions()
        self.assignIcons()
        self.fileExtensionMap = {'neuroML(*.xml *.nml)':'XML','csv(*.csv)':'CSV','HDF5(*.h5 *.hdf5)':'HDF5','All Supported(*.h5 *hdf5 *.xml *.nml *.csv)':'All'}
        self.fileBasedAction(['/home/chaitu/Desktop/GuiOG/Moogli/samples/mitral.h5'])
        

    def assignIcons(self):
        self.playButton.setIcon(QtGui.QIcon(os.path.join(PATH_ICONS,'play.png')))
        self.stopButton.setIcon(QtGui.QIcon(os.path.join(PATH_ICONS,'stop.png')))
        self.openToolButton.setIcon(QtGui.QIcon(os.path.join(PATH_ICONS,'open.png')))
        self.saveSnapButton.setIcon(QtGui.QIcon(os.path.join(PATH_ICONS,'saveSnap.png')))
        self.saveMovieButton.setIcon(QtGui.QIcon(os.path.join(PATH_ICONS,'saveMovie.png')))

    def clearCanvas(self):
        self.canvas.clearCanvas()
        self.canvas.updateGL()
        print 'Clearing Canvas'

    def connectActions(self):
        self.connect(self.actionQuit,QtCore.SIGNAL('triggered(bool)'),self.doQuit)
        self.connect(self.actionOpen,QtCore.SIGNAL('triggered(bool)'),self.openFile)
        self.connect(self.openToolButton,QtCore.SIGNAL('clicked()'),self.openFile)
        self.connect(self.canvas,QtCore.SIGNAL("compartmentSelected(QString)"),self.selectedCmpt)
        self.connect(self.actionPlay,QtCore.SIGNAL('triggered(bool)'),self.startTimer)
        self.connect(self.actionClearAll,QtCore.SIGNAL('triggered(bool)'),self.clearCanvas)
        self.connect(self.actionSaveAsMovie,QtCore.SIGNAL('triggered(bool)'),self.saveAsMovie)
        self.connect(self.actionSaveAsSnapshot,QtCore.SIGNAL('triggered(bool)'),self.saveAsSnapshot)
        self.connect(self.actionStop,QtCore.SIGNAL('triggered(bool)'),self.stopTimer)
        self.connect(self.saveSnapButton,QtCore.SIGNAL('clicked()'),self.saveAsSnapshot)
        self.connect(self.saveMovieButton,QtCore.SIGNAL('clicked()'),self.saveAsMovie)
        self.connect(self.plotToolButton,QtCore.SIGNAL('clicked()'),self.plottingOptions)


    def openFile(self):
        fileDialog = QtGui.QFileDialog(self)
        fileDialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        ffilter =''
        for key in sorted(self.fileExtensionMap.keys()):
            ffilter = ffilter + key + ';;'
        ffilter = ffilter[:-2]
        fileDialog.setFilter(self.tr(ffilter))
        fileDialog.setWindowTitle('Open File')

        targetPanel = QtGui.QFrame(fileDialog)
        targetPanel.setLayout(QtGui.QVBoxLayout())
        layout = fileDialog.layout()
        layout.addWidget(targetPanel)
        if fileDialog.exec_():
            self.clearCanvas() #clear all previous drawing
            fileNames = fileDialog.selectedFiles()
            self.fileBasedAction(fileNames)
        self.canvas.ctrlPressed = False #incase user uses ctrl+o to open file! - dirty code

    def fileBasedAction(self,fileNames):

        self.timeSlider.setEnabled(False)
        self.playButton.setEnabled(False)
        self.stopButton.setEnabled(False)
        self.plotToolButton.setEnabled(False)
        self.saveMovieButton.setEnabled(False)
        self.possibleData = False
        self.hasData = False

        for name in fileNames:
            fileType = str(name).rsplit('.',1)[1]
            fileName = str(name)
        print 'Opening File: ',fileName, 'Of Type', fileType
        self.parsedList = []
        if (fileType == 'xml') or (fileType =='nml'):
            from imports.MorphML import MorphML
            mml = MorphML()
            self.parsedList = mml.readMorphMLFromFile(fileName)
        elif (fileType == 'csv'):
            f = open(fileName,'r')
            testLine = f.readline()
            dialect = csv.Sniffer().sniff(testLine) #to get the format of the csv
            f.close()
            f = open(fileName, 'r')
            reader = csv.reader(f,dialect)
            for row in reader:
                self.parsedList.append([row[0],row[1],[float(i)*1e-6 for i in row[2:9]]])
            f.close()
        #elif (fileType == 'p'):
        elif (fileType == 'h5') or (fileType == 'hdf5'):
            self.dataFile = h5py.File(fileName)
            self.possibleData = True
            for name in self.dataFile.keys():
                if (name.find('.xml')!=-1) or (name.find('.nml')!=-1):
                    from imports.MorphML import MorphML
                    mml = MorphML()
                    self.parsedList = mml.readMorphMLFromString(str(self.dataFile[name].value[0]))
                elif (fileType == 'csv'):
                    f = open(fileName,'r')
                    testLine = f.readline()
                    dialect = csv.Sniffer().sniff(testLine) #to get the format of the csv
                    f.close()
                    f = open(fileName, 'r')
                    reader = csv.reader(f,dialect)
                    for row in reader:
                        self.parsedList.append([row[0],row[1],[float(i)*1e-6 for i in row[2:9]]])
                    f.close()

        else:
            print 'Not a supported Format yet*'

        if DEFAULT_DRAW:
            self.defaultDrawCompartments()
        else:
            self.selectCompartmentsToDraw()


    def checkForData(self):
        self.dataNameParaDict = {}
        for cmpt in self.parsedList:
            fullCmpt = cmpt[0]+'/'+cmpt[1]
            try:
                paras = self.dataFile[fullCmpt].keys()
                self.hasData = True
                self.dataNameParaDict[fullCmpt] = paras
                self.saveMovieButton.setEnabled(True)
            except KeyError: 
                pass
        
        self.cmptsVisualized = []

        if self.hasData:
            for fullCmptParaName in self.dataNameParaDict:
                try:
                    self.canvasStyleDict[fullCmptParaName] 
                    self.cmptsVisualized.append(fullCmptParaName)
                except KeyError:
                    pass
            self.statusbar.showMessage('You have data for '+str(len(self.dataNameParaDict))+' compartments.')# Of '+str(len(self.cmptsVisualized))+'/'+str(len(self.canvasStyleDict))+' compartments on Canvas')
            if DEFAULT_VISUALIZE:
                self.defaultVizCompartments()
            else:
                self.selectParameterToVisualize()
        else:
            print 'No data in h5 file / Are you naming them wrong?'


    def selectParameterToVisualize(self):
        self.colorMapLabelMapDict = {}
        self.attribColorMapLabelNameDict = {}
        self.attribNameColorLabelDict = {}

        self.vizCmptList = []
        self.vizCmptSelectedList = []

        self.newVizDia = QtGui.QDialog(self)
        self.paraSelectDia = paraSelectDia()
        self.paraSelectDia.setupUi(self.newVizDia)
        self.newVizDia.show()
        
        for cmpt in self.cmptsVisualized:
            for para in self.dataFile[cmpt].keys():
                self.paraSelectDia.listWidget.addItem(cmpt+'/'+para)
                self.vizCmptList.append(cmpt+'/'+para)

        #create default colormap
        self.defaultColorMap = colorMap(DEFAULT_COLORMAP_SELECTION,DEFAULT_COLORMAP_MINVAL,DEFAULT_COLORMAP_MAXVAL,DEFAULT_COLORMAP_LABEL)
        self.paraSelectDia.colorMapLabelComboBox.addItem(self.defaultColorMap.label)
        self.colorMapLabelMapDict[self.defaultColorMap.label] = self.defaultColorMap
        self.attribColorMapLabelNameDict[self.defaultColorMap.label] = []
        self.paraSelectDia.minDisplayPropertyLabel.setText(str(self.defaultColorMap.minVal))
        self.paraSelectDia.maxDisplayPropertyLabel.setText(str(self.defaultColorMap.maxVal))
        self.paraSelectDia.colorMapToolButton.setIcon(QtGui.QIcon(os.path.join(PATH_ICONS,'jet.png')))

        self.paraSelectDia.listWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.paraSelectDia.vizContentsListWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.connect(self.paraSelectDia.filterLineEdit,QtCore.SIGNAL('textEdited(QString)'),self.filterVisualizeList)
        self.connect(self.paraSelectDia.addToolButton,QtCore.SIGNAL('clicked()'),self.addToVizContentList)
        self.connect(self.paraSelectDia.minusToolButton,QtCore.SIGNAL('clicked()'),self.removeFromVizContentList)
        self.connect(self.paraSelectDia.allToolButton,QtCore.SIGNAL('clicked()'),self.addAllToVizContentList)
        self.connect(self.paraSelectDia.okayPushButton,QtCore.SIGNAL('clicked()'),self.addCompartmentsToViz)
        self.connect(self.paraSelectDia.useDefaultsPushButton,QtCore.SIGNAL('clicked()'),self.defaultVizCompartments)
        self.connect(self.paraSelectDia.editColorMapPushButton,QtCore.SIGNAL('clicked()'),self.editCurrentColorMap)
        self.connect(self.paraSelectDia.newColorMapPushButton,QtCore.SIGNAL('clicked()'),self.addNewColorMap)
        self.connect(self.paraSelectDia.colorMapLabelComboBox,QtCore.SIGNAL('currentIndexChanged(QString)'),self.refreshPropertyColorMap)

    def editCurrentColorMap(self):
        self.newColorMapDia = QtGui.QDialog(self)
        self.addColorMapDia = colorMapDia()
        self.addColorMapDia.setupUi(self.newColorMapDia)
        self.newColorMapDia.show()

        currentColorMap = self.colorMapLabelMapDict[str(self.paraSelectDia.colorMapLabelComboBox.currentText())]

        self.addColorMapDia.labelLineEdit.setText(currentColorMap.label)
        self.addColorMapDia.minValLineEdit.setText(str(currentColorMap.minVal))
        self.addColorMapDia.maxValLineEdit.setText(str(currentColorMap.maxVal))

        listOfAllFiles = os.listdir(PATH_COLORMAPS)
        for name in listOfAllFiles:
            if name == '.svn':
                listOfAllFiles.remove('.svn')
                pass
        self.addColorMapDia.colorMapComboBox.addItems(listOfAllFiles)

        for i in range(self.addColorMapDia.colorMapComboBox.count()):
            if self.addColorMapDia.colorMapComboBox.itemText(i) == str(currentColorMap.fileName):
                self.addColorMapDia.colorMapComboBox.setCurrentIndex(i)
        self.addColorMapDia.colorMapToolButton.setIcon(QtGui.QIcon(os.path.join(PATH_ICONS,str(self.addColorMapDia.colorMapComboBox.currentText())+'.png')))
        self.newColorMap = 0
        self.connect(self.addColorMapDia.okayPushButton,QtCore.SIGNAL('clicked()'),self.okayColorMap)
        self.connect(self.addColorMapDia.colorMapComboBox,QtCore.SIGNAL('currentIndexChanged(QString)'),self.updateColorMapToolDia)

    def refreshPropertyColorMap(self,name):
        name = str(name)
        self.paraSelectDia.minDisplayPropertyLabel.setText(str(self.colorMapLabelMapDict[name].minVal))
        self.paraSelectDia.maxDisplayPropertyLabel.setText(str(self.colorMapLabelMapDict[name].maxVal))
        self.paraSelectDia.colorMapToolButton.setIcon(QtGui.QIcon(os.path.join(PATH_ICONS,self.colorMapLabelMapDict[name].fileName+'.png')))

    def addNewColorMap(self):

        self.newColorMapDia = QtGui.QDialog(self)
        self.addColorMapDia = colorMapDia()
        self.addColorMapDia.setupUi(self.newColorMapDia)
        self.newColorMapDia.show()

        listOfAllFiles = os.listdir(PATH_COLORMAPS)
        for name in listOfAllFiles:
            if name == '.svn':
                listOfAllFiles.remove('.svn')
                pass

        self.newColorMap = 1
        self.addColorMapDia.colorMapComboBox.addItems(listOfAllFiles)
        self.addColorMapDia.colorMapToolButton.setIcon(QtGui.QIcon(os.path.join(PATH_ICONS,str(self.addColorMapDia.colorMapComboBox.currentText())+'.png')))
        self.connect(self.addColorMapDia.okayPushButton,QtCore.SIGNAL('clicked()'),self.okayColorMap)
        self.connect(self.addColorMapDia.colorMapComboBox,QtCore.SIGNAL('currentIndexChanged(QString)'),self.updateColorMapToolDia)

    def updateColorMapToolDia(self,name):
        self.addColorMapDia.colorMapToolButton.setIcon(QtGui.QIcon(os.path.join(PATH_ICONS,str(name)+'.png')))

    def okayColorMap(self):
        if float(self.addColorMapDia.maxValLineEdit.text())<= float(self.addColorMapDia.minValLineEdit.text()):
            print 'inValid, Max <= Min'
        else:
            if (self.newColorMap): #case via add new colormap
                newcMap = colorMap(str(self.addColorMapDia.colorMapComboBox.itemText(self.addColorMapDia.colorMapComboBox.currentIndex())),float(self.addColorMapDia.minValLineEdit.text()),float(self.addColorMapDia.maxValLineEdit.text()),str(self.addColorMapDia.labelLineEdit.text()))
                self.colorMapLabelMapDict[str(self.addColorMapDia.labelLineEdit.text())] = newcMap
                self.paraSelectDia.colorMapLabelComboBox.addItem(str(self.addColorMapDia.labelLineEdit.text()))
                self.attribColorMapLabelNameDict[str(self.addColorMapDia.labelLineEdit.text())] =[]
            else: #case via edit a colormap
                currentColorMap = self.colorMapLabelMapDict[str(self.addColorMapDia.labelLineEdit.text())]
                currentColorMap.setColorMap(fileName=str(self.addColorMapDia.colorMapComboBox.itemText(self.addColorMapDia.colorMapComboBox.currentIndex())),label=str(self.addColorMapDia.labelLineEdit.text()),minVal=float(self.addColorMapDia.minValLineEdit.text()),maxVal=float(self.addColorMapDia.maxValLineEdit.text()))

            self.newColorMapDia.close()

            for index in range(self.paraSelectDia.colorMapLabelComboBox.count()):
                if self.paraSelectDia.colorMapLabelComboBox.itemText(index) == str(self.addColorMapDia.labelLineEdit.text()):
                    self.paraSelectDia.colorMapLabelComboBox.setCurrentIndex(index)
                    pass
        self.refreshPropertyColorMap(str(self.paraSelectDia.colorMapLabelComboBox.currentText()))

    def filterVisualizeList(self,string):
        self.paraSelectDia.listWidget.clear()
        for names in self.vizCmptList:
            if names.find(string) != -1:
                self.paraSelectDia.listWidget.addItem(names)

    def filterSelectionList(self,string): 
        self.styleSelectDia.listWidget.clear()
        for name in self.cmptDict:
            if name.find(string) != -1:
                self.styleSelectDia.listWidget.addItem(name)

    def selectCompartmentsToDraw(self):
        self.cmptDict = {}
        self.canvasStyleDict = {}
        self.attribStyleDict={}
        
        self.newStyleDia = QtGui.QDialog(self)
        self.styleSelectDia = styleSelectDia()
        self.styleSelectDia.setupUi(self.newStyleDia)
        self.newStyleDia.show()
        
        self.styleSelectDia.listWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.styleSelectDia.canvasContentsListWidget.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.connect(self.styleSelectDia.filterLineEdit,QtCore.SIGNAL('textEdited(QString)'),self.filterSelectionList)
        self.connect(self.styleSelectDia.addToolButton,QtCore.SIGNAL('clicked()'),self.addToCanvasContentList)
        self.connect(self.styleSelectDia.minusToolButton,QtCore.SIGNAL('clicked()'),self.removeFromCanvasContentList)
        self.connect(self.styleSelectDia.allToolButton,QtCore.SIGNAL('clicked()'),self.addAllToCanvasContentList)
        self.connect(self.styleSelectDia.okayPushButton,QtCore.SIGNAL('clicked()'),self.drawCompartments)
        self.connect(self.styleSelectDia.useDefaultsPushButton,QtCore.SIGNAL('clicked()'),self.defaultDrawCompartments)
        
       #reparse parsedlist ?

        for j,i in enumerate(self.parsedList):
            self.styleSelectDia.listWidget.addItem(i[0]+'/'+i[1])
            self.cmptDict[i[0]+'/'+i[1]] = j

    def defaultDrawCompartments(self):
        print 'Default Draw Compartment'
        self.canvasStyleDict = {}
        self.attribStyleDict = {}
        for index,cmpt in enumerate(self.parsedList): #include cases for copy cell, and default offset
            self.canvas.drawNewCompartment(cmpt[0],cmpt[1],[i*1e+04 for i in cmpt[2][:7]],style=DEFAULT_DRAW_STYLE)
            self.canvasStyleDict[cmpt[0]+'/'+cmpt[1]] = index
            self.attribStyleDict[cmpt[0]+'/'+cmpt[1]] = [DEFAULT_DRAW_STYLE,'0.0,0.0,0.0','0.0,0.0,0.0,0.0'] 
        self.canvas.updateGL()

        if not DEFAULT_DRAW: 
            self.newStyleDia.close() #close the dialog

        if self.possibleData:
            self.checkForData()

    def addToCanvasContentList(self):
        for items in self.styleSelectDia.listWidget.selectedItems():
            self.styleSelectDia.canvasContentsListWidget.addItem(str(items.text()+',style='+self.styleSelectDia.styleComboBox.currentText()+',offsetCentre=['+self.styleSelectDia.offsetPositionLineEdit.text()+'],offsetAngle=['+self.styleSelectDia.offsetAngleLineEdit.text())+']')
            self.canvasStyleDict[str(items.text())] = self.cmptDict.pop(str(items.text())) 
            self.attribStyleDict[str(items.text())] = [1+self.styleSelectDia.styleComboBox.currentIndex(),self.styleSelectDia.offsetPositionLineEdit.text(),self.styleSelectDia.offsetAngleLineEdit.text()]
        self.filterSelectionList(self.styleSelectDia.filterLineEdit.text())

    def addAllToCanvasContentList(self):
        self.styleSelectDia.listWidget.selectAll()
        self.addToCanvasContentList()

    def removeFromCanvasContentList(self):
        for items in self.styleSelectDia.canvasContentsListWidget.selectedItems():
            nameCellCmpt = str(items.text()).split(',',1)[0]
            self.styleSelectDia.listWidget.addItem(nameCellCmpt)
            self.cmptDict[nameCellCmpt] = self.canvasStyleDict.pop(nameCellCmpt)
           
            self.styleSelectDia.canvasContentsListWidget.takeItem(self.styleSelectDia.canvasContentsListWidget.indexFromItem(items).row())
            self.attribStyleDict.pop(nameCellCmpt)

        self.filterSelectionList(self.styleSelectDia.filterLineEdit.text())

    def drawCompartments(self):
        for number in range(self.styleSelectDia.canvasContentsListWidget.count()):
            item = self.styleSelectDia.canvasContentsListWidget.item(number)
            nameCellCmpt = str(item.text()).split(',',1)[0]

            index = self.canvasStyleDict[nameCellCmpt]
            attribStyle = self.attribStyleDict[nameCellCmpt]

            self.canvas.drawNewCompartment(self.parsedList[index][0],self.parsedList[index][1],[i*1e+04 for i in self.parsedList[index][2][:7]],style=attribStyle[0],cellCentre=[float(i) for i in attribStyle[1].split(',')],cellAngle=[float(i) for i in attribStyle[2].split(',')])

        self.canvas.updateGL()
        self.newStyleDia.close()

        if self.possibleData:
            self.checkForData()

    def defaultVizCompartments(self):
        print 'Visualizing with defaults'
        self.vizCmptSelectedList = []
        self.attribColorMapLabelNameDict = {}
        self.attribNameColorLabelDict = {}

        if DEFAULT_VISUALIZE:
            self.colorMapLabelMapDict = {}
            self.defaultColorMap = colorMap(DEFAULT_COLORMAP_SELECTION,DEFAULT_COLORMAP_MINVAL,DEFAULT_COLORMAP_MAXVAL,DEFAULT_COLORMAP_LABEL)
            self.colorMapLabelMapDict[self.defaultColorMap.label] = self.defaultColorMap

        for cmpt in self.cmptsVisualized:
            for para in self.dataFile[cmpt].keys():
                self.vizCmptSelectedList.append(cmpt+'/'+para)
                self.attribNameColorLabelDict[cmpt+'/'+para] = self.defaultColorMap.label

        self.attribColorMapLabelNameDict[self.defaultColorMap.label] = self.vizCmptSelectedList

        self.addCompartmentsToViz()

    def addToVizContentList(self):
        for items in self.paraSelectDia.listWidget.selectedItems():
            self.paraSelectDia.vizContentsListWidget.addItem(str(items.text())+',colorMap=' +str(self.paraSelectDia.colorMapLabelComboBox.currentText()))
            self.vizCmptSelectedList.append(str(items.text()))
            self.vizCmptList.remove(str(items.text()))
            self.attribColorMapLabelNameDict[str(self.paraSelectDia.colorMapLabelComboBox.currentText())].append(str(items.text()))
            self.attribNameColorLabelDict[str(items.text())] = str(self.paraSelectDia.colorMapLabelComboBox.currentText()) 
        self.filterVisualizeList(self.paraSelectDia.filterLineEdit.text())

    def addAllToVizContentList(self):
        self.paraSelectDia.listWidget.selectAll()
        self.addToVizContentList()

    def removeFromVizContentList(self):
        for items in self.paraSelectDia.vizContentsListWidget.selectedItems():
            nameCellCmptPara = str(items.text()).split(',',1)[0]
            self.paraSelectDia.listWidget.addItem(nameCellCmptPara)
            self.vizCmptList.append(nameCellCmptPara)
            self.vizCmptSelectedList.remove(nameCellCmptPara)
            self.attribColorMapLabelNameDict[self.attribNameColorLabelDict.pop(nameCellCmptPara)].remove(nameCellCmptPara)
            self.paraSelectDia.vizContentsListWidget.takeItem(self.paraSelectDia.vizContentsListWidget.indexFromItem(items).row())
        self.filterVisualizeList(self.paraSelectDia.filterLineEdit.text())

    def addCompartmentsToViz(self):
#        print self.attribColorMapLabelNameDict,self.attribNameColorLabelDict
        self.cmptNameValuesDict = {}
        for cmpt in self.vizCmptSelectedList:
            self.cmptNameValuesDict[cmpt] = self.dataFile[cmpt].value
        for value in self.cmptNameValuesDict:
            try:
                if len(self.cmptNameValuesDict[value]) < self.numberDataPoints :
                    self.numberDataPoints = len(self.cmptNameValuesDict[value])
                else:
                    pass
            except AttributeError:
                self.numberDataPoints = len(self.cmptNameValuesDict[value])

        if not DEFAULT_VISUALIZE:
            self.newVizDia.close()

        self.binProperties()

    def binProperties(self):
        if not DEFAULT_BIN:
            self.newBinPropDia = QtGui.QDialog(self)
            self.binPropDia = binPropertyDia()
            self.binPropDia.setupUi(self.newBinPropDia)
            self.newBinPropDia.show()
            
            self.binPropDia.numberOfDataPointsLabel.setText(str(self.numberDataPoints))
            self.refreshNumberOfFrames(self.numberDataPoints)

            self.connect(self.binPropDia.skipFramesRadioButton,QtCore.SIGNAL('clicked()'),self.activateBinSize)
            self.connect(self.binPropDia.binMeanRadioButton,QtCore.SIGNAL('clicked()'),self.activateBinSize)
            self.connect(self.binPropDia.binMaxRadioButton,QtCore.SIGNAL('clicked()'),self.activateBinSize)
            self.connect(self.binPropDia.noBinRadioButton,QtCore.SIGNAL('clicked()'),self.deactivateBinSize)
            self.connect(self.binPropDia.binSizeLineEdit,QtCore.SIGNAL('textEdited(QString)'),self.refreshNumberOfFrames)
            self.connect(self.binPropDia.useDefaultPushButton,QtCore.SIGNAL('clicked()'),self.useDefaultBin)
            self.connect(self.binPropDia.okayPushButton,QtCore.SIGNAL('clicked()'),self.binSelection)
        else:
            self.useDefaultBin()

    def binSelection(self):
        if self.binPropDia.noBinRadioButton.isChecked():
            self.binMode = 1
        elif self.binPropDia.skipFramesRadioButton.isChecked():
            self.binMode = 2
        elif self.binPropDia.binMeanRadioButton.isChecked():
            self.binMode = 3
        else:
            self.binMode = 4

        if self.binMode!=1:
            self.binSize = int(self.binPropDia.binSizeLineEdit.text())
        else:
            self.binSize = 1
        self.numberOfFrames = int(self.binPropDia.numberOfFramesLabel.text())
        self.newBinPropDia.close()
        self.doBin()

    def useDefaultBin(self):
        if not DEFAULT_BIN:
            self.newBinPropDia.close()
        print 'Default Binning Properties'
        self.binMode = DEFAULT_BIN_MODE
        if self.binMode != 1:
            self.numberOfFrames = int(self.numberDataPoints / DEFAULT_BIN_SIZE)
        else:#nobin case
            self.numberOfFrames = self.numberDataPoints
        self.binSize = DEFAULT_BIN_SIZE
        self.doBin()

    def doBin(self):
        self.statusbar.showMessage(str(str(self.numberDataPoints) +' datapoints shown in '+str(self.numberOfFrames) +' frames'))
        self.frameData = {}
        if self.binMode == 1: #nobin
            self.frameData = self.cmptNameValuesDict
        elif self.binMode == 2: #skipFrames
            for cmptParaName in self.cmptNameValuesDict:
                self.frameData[cmptParaName] = []
                for i in range(0,self.numberDataPoints,self.binSize):
                    self.frameData[cmptParaName].append(self.cmptNameValuesDict[cmptParaName][i])
        elif self.binMode ==3: #meanOfBinFrames
            for cmptParaName in self.cmptNameValuesDict:
                self.frameData[cmptParaName] = []
                for i in range(0,self.numberDataPoints,self.binSize):
                    self.frameData[cmptParaName].append((self.cmptNameValuesDict[cmptParaName][i:min((i+self.binSize),self.numberDataPoints)].mean()))
        else: #maxOfBinFrames
            for cmptParaName in self.cmptNameValuesDict:
                self.frameData[cmptParaName] = []
                for i in range(0,self.numberDataPoints,self.binSize):
                    self.frameData[cmptParaName].append(self.cmptNameValuesDict[cmptParaName][i:min((i+self.binSize),self.numberDataPoints)].max())
        
        self.setupForViz()

    def setupForViz(self):
        self.playButton.setEnabled(True)
        self.stopButton.setEnabled(True)

        self.timeSlider.setEnabled(True)
        self.timeSlider.setRange(0,self.numberOfFrames-1)
        self.timeSlider.setTickPosition(0)
        self.ctimer = QtCore.QTimer()

        self.connect(self.ctimer, QtCore.SIGNAL("timeout()"),self.playMovie)
        self.connect(self.playButton, QtCore.SIGNAL('clicked()'),self.startTimer)
        self.connect(self.stopButton, QtCore.SIGNAL('clicked()'),self.stopTimer)
        self.connect(self.timeSlider, QtCore.SIGNAL('valueChanged(int)'), self.getValue)

        self.addCmptToVizCanvas()
        self.digitizeCMapIndex()

    def addCmptToVizCanvas(self):
        for item in self.vizCmptSelectedList:
            self.canvas.addToVisualize(item.rsplit('/',1)[0])

    def digitizeCMapIndex(self):
        for cMapLabel in self.attribColorMapLabelNameDict:
            for cmptParaName in self.attribColorMapLabelNameDict[cMapLabel]:
                self.frameData[cmptParaName] = numpy.digitize(self.frameData[cmptParaName],self.colorMapLabelMapDict[cMapLabel].stepVals)

    def updateViz(self,value):
        for cMapLabel in self.attribColorMapLabelNameDict:
            for cmptParaName in self.attribColorMapLabelNameDict[cMapLabel]:
                thisCmpt = cmptParaName.rsplit('/',1)[0]
                self.canvas.vizObjects[thisCmpt].r,self.canvas.vizObjects[thisCmpt].g,self.canvas.vizObjects[thisCmpt].b = self.colorMapLabelMapDict[cMapLabel].colorMap[self.frameData[cmptParaName][value]]
        self.canvas.updateGL()

    def saveAsMovie(self):
        #dialog asks for which frame to whicheth frame goes here.
        self.movieSaveInputDialog = QtGui.QInputDialog()
        text,pressOK =  self.movieSaveInputDialog.getText(self.movieSaveInputDialog,'Save Frames(Movie)','First,Last:')
        if pressOK:
            frameNumbers = str(text).split(',')
            if len(frameNumbers) == 2 :
                if (int(frameNumbers[0]) < int(frameNumbers[1])):
                    self.saveAsAvi(int(frameNumbers[0]),int(frameNumbers[1]))
                else:
                    print 'starting frame > ending frame'
            else:
                print 'wrong inputs'
 
    def saveAsAvi(self,start,stop):
        oldPath = os.getcwd()

        if DEFAULT_SAVE:
            file_location = DEFAULT_FILESAVE_LOCATION
            file_name = DEFAULT_MOVIE_FILENAME
        else:
            fileDialog = QtGui.QFileDialog(self)        
            file_fullPath = str(fileDialog.getSaveFileName(self,'Save As',".avi"))
            if not file_fullPath=='':
                file_location = os.path.split(file_fullPath)[0]
                file_name = os.path.split(file_fullPath)[1]
            else:
                return

        if not os.path.exists(os.path.join(PATH_MAIN,'movieMake')):
            os.mkdir(os.path.join(PATH_MAIN,'movieMake'))

        os.chdir(os.path.join(PATH_MAIN,'movieMake'))

        for framenum in range(start,stop+1):
            self.timeSlider.setValue(framenum)
            pic = self.canvas.grabFrameBuffer()
            pic.save('sim_'+str(framenum)+'.png','PNG')

        f = open('filelist.txt','w')
        filelist = ["sim_"+str(i)+".png" for i in range(start,stop+1)]
        f.write('\n'.join(filelist))
        f.close()

    ## mpeg4 compression
        mpegString = 'mencoder mf://@filelist.txt -mf w='+str(DEFAULT_MOVIE_WIDTH)+':h='+str(DEFAULT_MOVIE_HEIGHT)+':fps='+str(DEFAULT_MOVIE_FPS)+':type=png  -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o '+ file_name
        os.system(mpegString)

        shutil.move(os.path.join(PATH_MAIN,'movieMake',file_name),file_location)
        os.path.abspath(oldPath)

    def makeColorMapPNG(fileName):
        f = open(fileName,'r')
        l = pickle.load(f)
        width = len(l)
        height = 20
        a = numpy.zeros([height,width,3])
        
        for i in range(height):
            for j in range(width):
                for k in range(3):
                    a[i][j][k] = l[j][k]

        plt.imshow(a)
        plt.xticks([])
        plt.yticks([])
        plt.set_alpha(0)
        plt.savefig(os.path.split(fileName)[1]+'.png',transparent='True',bbox_inches='tight',dpi=80)
        shutil.move(fileName+'.png',PATH_ICONS)

    def saveAsSnapshot(self):
        pic = self.canvas.grabFrameBuffer()
        if DEFAULT_SAVE:
            pic.save(os.path.join(DEFAULT_FILESAVE_LOCATION,DEFAULT_SNAPSHOT_FILENAME),'PNG')
        else:
            fileDialog = QtGui.QFileDialog(self)        
            file_fullPath = str(fileDialog.getSaveFileName(self,'Save As',".png"))
            if not file_fullPath=='':
                pic.save(file_fullPath,'PNG')

    def selectedCmpt(self,name):
        self.statusbar.showMessage(name)
        if self.hasData and name !='': #activate plot button
            try:
                self.dataFile[str(name)]
                self.plotToolButton.setEnabled(True)

            except KeyError:
                self.plotToolButton.setEnabled(False)
                pass
        else:
            self.plotToolButton.setEnabled(False)

    def plottingOptions(self): #pick selected from canvas
        allSelectedNamesParaDict = {}
        for selected in self.canvas.selectedObjects:
            try:
                allSelectedNamesParaDict[selected.daddy+'/'+selected.name] = self.dataFile[selected.daddy+'/'+selected.name].keys()
            except KeyError:
                pass

        #prep for plotting
        newPlotDia = QtGui.QMainWindow(self) #gimmic - need to call matplotlib inside qt,else everthing freezes! also the navigation bar for nice-ness
        newPlotWin = gPlotWin()
        newPlotWin.setupUi(newPlotDia)
        qToolBar = QtGui.QToolBar()
        cToolbar = NavigationToolbar(newPlotWin.plotCanvas, qToolBar)
        qToolBar.addWidget(cToolbar)
        qToolBar.setMovable(False)
        qToolBar.setFloatable(False)
        newPlotDia.addToolBar(QtCore.Qt.BottomToolBarArea,qToolBar)
        newPlotDia.show()

        for cmptName in allSelectedNamesParaDict:
            if len(allSelectedNamesParaDict[cmptName]) == 1:
                line = newPlotWin.plotCanvas.update_graph(self.dataFile[cmptName+'/'+allSelectedNamesParaDict[cmptName][0]].value,cmptName+'/'+allSelectedNamesParaDict[cmptName][0])
                newPlotWin.plotCanvas.axes.set_ylabel(allSelectedNamesParaDict[cmptName][0])
            else:
#                print 'multiple parameters to plot. Select One.'
                self.plotParaInputDialog = QtGui.QInputDialog()
                paraName,pressOK =  self.plotParaInputDialog.getText(self.plotParaInputDialog,'Select Parameter to Plot:',str(allSelectedNamesParaDict[cmptName]))
                if pressOK:
                    paraName = str(paraName)
                    try:
                        paraIndex = allSelectedNamesParaDict[cmptName].index(paraName)
                        line = newPlotWin.plotCanvas.update_graph(self.dataFile[cmptName+'/'+allSelectedNamesParaDict[cmptName][0]].value,cmptName+'/'+allSelectedNamesParaDict[cmptName][paraIndex])
                        newPlotWin.plotCanvas.axes.set_ylabel(allSelectedNamesParaDict[cmptName][paraIndex])
                    except ValueError:
                        print 'Invalid parameter name'

    def playMovie(self):
        self.playButton.setIcon(QtGui.QIcon(os.path.join(PATH_ICONS,'pause.png')))
        t = self.timeSlider.value()
        t += 1        
        if t < self.numberOfFrames:
            self.timeSlider.setValue(t)
        else:
            self.stopTimer()

    def getValue(self,value):
        self.updateViz(value)
        self.statusbar.showMessage(str('Showing '+ str(value)+ '/' +str(self.numberOfFrames-1)))

    def startTimer(self):
        if self.ctimer.isActive():
            self.ctimer.stop()
            self.playButton.setIcon(QtGui.QIcon(os.path.join(PATH_ICONS,'play.png')))
        else:
            self.ctimer.start(DEFAULT_FRAMEUPDATEt)

    def stopTimer(self):
        self.ctimer.stop()
        self.timeSlider.setValue(0)
        self.playButton.setIcon(QtGui.QIcon(os.path.join(PATH_ICONS,'play.png')))

    def refreshNumberOfFrames(self,text):
        if text !='':
            try:
                text = int(text)
                if (text <= 0) :
                    print 'Division by Zero. Minimum is 1.'
                    self.binPropDia.binSizeLineEdit.setText(str(1))
                if self.binPropDia.binSizeLabel.isEnabled():
                    self.binPropDia.numberOfFramesLabel.setText(str(int(self.numberDataPoints/int(self.binPropDia.binSizeLineEdit.text()))))
                else:
                    self.binPropDia.numberOfFramesLabel.setText(str(self.numberDataPoints))
            except ValueError:
                print 'Enter valid whole number not text for bin size'

    def activateBinSize(self):
        self.binPropDia.binSizeLabel.setEnabled(True)
        self.binPropDia.binSizeLineEdit.setEnabled(True)
        self.binPropDia.binSizeLineEdit.setText(str(DEFAULT_BIN_SIZE))
        self.refreshNumberOfFrames(str(DEFAULT_BIN_SIZE))

    def deactivateBinSize(self):
        self.binPropDia.binSizeLabel.setEnabled(False)
        self.binPropDia.binSizeLineEdit.setEnabled(False)
        self.refreshNumberOfFrames(str(self.binPropDia.numberOfDataPointsLabel.text()))
       
    def doQuit(self):
        QtGui.qApp.closeAllWindows()

app = QtGui.QApplication(sys.argv)
dmw = DesignerMainWindow()
dmw.show()
sys.exit(app.exec_())
