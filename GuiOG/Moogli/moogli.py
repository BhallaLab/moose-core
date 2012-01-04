import sys
import os

from canvas import *
from moogliLayout import *
from defaults import *
from layout.styleSelector import Ui_Dialog as styleSelectDia
from layout.vizSelector import Ui_Dialog as paraSelectDia
from layout.colorMap import Ui_Dialog as colorMapDia
from PyQt4 import Qt,QtGui,QtCore

class DesignerMainWindow(QtGui.QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(DesignerMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.connectActions()
        self.fileExtensionMap = {'neuroML(*.xml *.nml)':'XML','csv(*.csv)':'CSV','HDF5(*.h5 *.hdf5)':'HDF5','All Supported(*.h5 *hdf5 *.xml *.nml *.csv)':'All'}
        self.fileBasedAction(['/home/chaitu/Desktop/GuiOG/Moogli/samples/mitral.h5'])
        self.possibleData = False
        self.hasData = False

    def clearCanvas(self):
        self.canvas.clearCanvas()
        print 'Clearing Canvas'

    def connectActions(self):
        self.connect(self.actionQuit,QtCore.SIGNAL('triggered(bool)'),self.doQuit)
        self.connect(self.actionOpen,QtCore.SIGNAL('triggered(bool)'),self.openFile)
        self.connect(self.canvas,QtCore.SIGNAL("compartmentSelected(QString)"),self.selectedCmpt)

    def selectedCmpt(self,name):
        self.statusbar.showMessage(name)

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
        for name in fileNames:
            fileType = str(name).rsplit('.',1)[1]
            fileName = str(name)
        print 'Opening File: ',fileName, 'Of Type', fileType
        self.parsedList = []
        if (fileType == 'xml') or (fileType =='nml'):
            from imports.MorphML import MorphML
            mml = MorphML()
            self.parsedList = mml.readMorphMLFromFile(fileName)
        #elif (fileType == 'csv'):
        #elif (fileType == 'p'):
        elif (fileType == 'h5') or (fileType == 'hdf5'):
            import h5py
            h5file = h5py.File(fileName)
            self.possibleData = True
            for name in h5file.keys():
                if (name.find('.xml')!=-1) or (name.find('.nml')!=-1):
                    from imports.MorphML import MorphML
                    mml = MorphML()
                    self.parsedList = mml.readMorphMLFromString(str(h5file[name].value[0]))
        else:
            print 'Not a supported Format yet*'

        if DEFAULT_DRAW:
            print 'Default draw of Compartments'
            self.defaultDrawCompartments()
        else:
            self.selectCompartmentsToDraw()

        if self.possibleData:
            self.checkForData(h5file)

    def checkForData(self,h5file):
        self.dataNameParaDict = {}
        for cmpt in self.parsedList:
            fullCmpt = cmpt[1]+'/'+cmpt[0]
            try:
                paras = h5file[fullCmpt].keys()
                self.hasData = True
                self.dataNameParaDict[fullCmpt] = paras
            except KeyError: 
                pass
        
        self.cmptsVisualized = []

        if self.hasData:
            self.dataFile = h5file
            for fullCmptParaName in self.dataNameParaDict:
                try:
                    self.canvasStyleDict[fullCmptParaName] 
                    self.cmptsVisualized.append(fullCmptParaName)
                except KeyError:
                    pass
            self.statusbar.showMessage('You have data for '+str(len(self.dataNameParaDict))+' compartments.')# Of '+str(len(self.cmptsVisualized))+'/'+str(len(self.canvasStyleDict))+' compartments on Canvas')

            if DEFAULT_VISUALIZE:
                print 'Visualizing with defaults'
                self.defaultVizCompartments()
            else:
                self.selectParameterToVisualize(h5file)
        else:
            print 'No data in h5 file / Are you naming them wrong?'


    def selectParameterToVisualize(self,h5file):
        self.colorMapLabelMapDict = {}

        self.vizCmptList = []
        self.vizCmptSelectedList = []
        self.attribColorMapLabelNameDict = {}
        self.attribNameColorLabelDict = {}

        self.newVizDia = QtGui.QDialog(self)
        self.paraSelectDia = paraSelectDia()
        self.paraSelectDia.setupUi(self.newVizDia)
        self.newVizDia.show()
        
        for cmpt in self.cmptsVisualized:
            for para in h5file[cmpt].keys():
                self.paraSelectDia.listWidget.addItem(cmpt+'/'+para)
                self.vizCmptList.append(cmpt+'/'+para)

        #create default colormap
        self.defaultColorMap = colorMap(DEFAULT_COLORMAP_SELECTION,DEFAULT_COLORMAP_MINVAL,DEFAULT_COLORMAP_MAXVAL,DEFAULT_COLORMAP_LABEL)
        self.paraSelectDia.colorMapLabelComboBox.addItem(self.defaultColorMap.label)
        self.colorMapLabelMapDict[self.defaultColorMap.label] = self.defaultColorMap
        self.attribColorMapLabelNameDict[self.defaultColorMap.label] = []
        self.paraSelectDia.displayPropertyLabel.setText('MinValue='+str(self.defaultColorMap.minVal)+',MaxVal='+str(self.defaultColorMap.maxVal)+',ColorMap='+self.defaultColorMap.fileName)

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
        self.newColorMap = 0
        self.connect(self.addColorMapDia.okayPushButton,QtCore.SIGNAL('clicked()'),self.okayColorMap)


    def refreshPropertyColorMap(self,name):
        name = str(name)
        self.paraSelectDia.displayPropertyLabel.setText('MinValue='+str(self.colorMapLabelMapDict[name].minVal)+',MaxVal='+str(self.colorMapLabelMapDict[name].maxVal)+',ColorMap='+self.colorMapLabelMapDict[name].fileName)

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
        self.connect(self.addColorMapDia.okayPushButton,QtCore.SIGNAL('clicked()'),self.okayColorMap)

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

        for j,i in enumerate(self.parsedList):
            self.styleSelectDia.listWidget.addItem(i[1]+'/'+i[0])
            self.cmptDict[i[1]+'/'+i[0]] = j

    def defaultDrawCompartments(self):
        self.canvasStyleDict = {}
        self.attribStyleDict = {}
        for index,cmpt in enumerate(self.parsedList):
            self.canvas.drawNewCompartment(cmpt[0],cmpt[1],[i*1e+04 for i in cmpt[2][:7]],style=DEFAULT_DRAW_STYLE)
            self.canvasStyleDict[cmpt[1]+'/'+cmpt[0]] = index
            self.attribStyleDict[cmpt[1]+'/'+cmpt[0]] = [DEFAULT_DRAW_STYLE,'0.0,0.0,0.0','0.0,0.0,0.0,0.0'] #by default all are drawn at the origin with zero angle, do not change this
        self.canvas.updateGL()

        if not DEFAULT_DRAW: 
            self.newStyleDia.close() #close the dialog

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

    def defaultVizCompartments(self):
        self.vizCmptSelectedList = []
        self.attribColorMapLabelNameDict = {}
        self.attribNameColorLabelDict = {}

        if DEFAULT_VISUALIZE:
            self.colorMapLabelMapDict = {}
            self.defaultColorMap = colorMap(DEFAULT_COLORMAP_SELECTION,DEFAULT_COLORMAP_MINVAL,DEFAULT_COLORMAP_MAXVAL,DEFAULT_COLORMAP_LABEL)
            self.colorMapLabelMapDict[self.defaultColorMap.label] = self.defaultColorMap

        h5file = self.dataFile
        for cmpt in self.cmptsVisualized:
            for para in h5file[cmpt].keys():
                self.vizCmptSelectedList.append(cmpt+'/'+para)
                self.attribNameColorLabelDict[cmpt+'/'+para] = self.defaultColorMap.label

        self.attribColorMapLabelNameDict[self.defaultColorMap.label] = self.vizCmptSelectedList
                
        if not DEFAULT_VISUALIZE:
            self.newVizDia.close()

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
        print self.attribColorMapLabelNameDict,self.attribNameColorLabelDict
        
    def doQuit(self):
        QtGui.qApp.closeAllWindows()

app = QtGui.QApplication(sys.argv)
dmw = DesignerMainWindow()
dmw.show()
sys.exit(app.exec_())
