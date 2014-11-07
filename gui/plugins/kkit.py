import sys
#import math
#import re
from PyQt4 import QtGui, QtCore, Qt
#import pygraphviz as pgv
import networkx as nx
#sys.path.insert(0, '~/async/gui')
#import numpy as np
from default import *
from moose import *
#sys.path.append('plugins')
from mplugin import *
from kkitUtil import *
from kkitQGraphics import PoolItem, ReacItem,EnzItem,CplxItem,ComptItem
from kkitViewcontrol import *
from kkitCalcArrow import *
from kkitOrdinateUtil import *
import posixpath
from mtoolbutton import MToolButton
from PyQt4.QtGui import QWidget
from PyQt4.QtGui import QGridLayout
from PyQt4.QtGui import QColor
import RunWidget

#from DataTable import DataTable
class KkitPlugin(MoosePlugin):
    """Default plugin for MOOSE GUI"""
    def __init__(self, *args):
        #print args
        MoosePlugin.__init__(self, *args)
        self.view = None
        #self.plotView = PlotView(self)
        #self.getRunView()
        #self.plotView.dataTable = self.view._centralWidget.dataTable
        #self.plotView.updateCallback = self.view._centralWidget.legendUpdate
        #self.view._centralWidget.legendUpdate()
        #self.dataTable = DataTable(self.dataRoot)


    def getPreviousPlugin(self):
        return None

    def getNextPlugin(self):
        return None

    def getAdjacentPlugins(self):
        return []

    def getViews(self):
        return self._views

    def getCurrentView(self):
        return self.currentView

    def getEditorView(self):
        if not hasattr(self, 'editorView'):
            #self.editorView = KkitEditorView(self, self.dataTable)
            self.editorView = KkitEditorView(self)
            self.editorView.getCentralWidget().editObject.connect(self.mainWindow.objectEditSlot)
            #self.editorView.GrViewresize(self)
            #self.editorView.connect(self,QtCore.SIGNAL("resize(QResizeEvent)"),self.editorView.GrViewresize)
            self.currentView = self.editorView
        return self.editorView

    def getRunView(self):
        if self.view is None:
            self.view = AnotherKkitRunView(self.modelRoot, self)
        return self.view


        if self.view is not None: return AnotherKkitRunView(self.modelRoot, self)
        if self.view is not None: return self.view
        self.view = RunView(self.modelRoot, self)
        graphView = self.view._centralWidget
        graphView.setDataRoot(self.modelRoot)
        graphView.plotAllData()
        schedulingDockWidget = self.view.getSchedulingDockWidget().widget()
        self._kkitWidget = self.view.plugin.getEditorView().getCentralWidget()
        #self.runView = KkitRunView(self,self.dataTable)
        self.runView = KkitRunView(self)
        self.currentRunView = self.runView.getCentralWidget()

        #schedulingDockWidget.runner.update.connect(self.currentRunView.changeBgSize)
        #schedulingDockWidget.runner.resetAndRun.connect(self.currentRunView.resetColor)
        graphView.layout().addWidget(self.currentRunView,0,0,2,1)
        return self.view

# class AnotherKkitRunViewsCentralWidget(QWidget):

#     def __init__():
#         QWidget.__init__()

#     def

class AnotherKkitRunView(RunView):

    def __init__(self, modelRoot, plugin,*args):
        RunView.__init__(self, modelRoot, plugin, *args)
        self.modelRoot = modelRoot
        self.plugin    = plugin
        self.schedular = None

    def createCentralWidget(self):
        self._centralWidget = RunWidget.RunWidget(self.modelRoot)
        self.kkitRunView   = KkitRunView(self.plugin)
        self.plotWidgetContainer = PlotWidgetContainer(self.modelRoot)
        self._centralWidget.setChildWidget(self.kkitRunView.getCentralWidget(), False, 0, 0, 1, 1)
        self._centralWidget.setChildWidget(self.plotWidgetContainer, False, 0, 1, 1, 2)
        self._centralWidget.setPlotWidgetContainer(self.plotWidgetContainer)
        self.schedular = self.getSchedulingDockWidget().widget()
        self.schedular.runner.simulationProgressed.connect(self.kkitRunView.getCentralWidget().updateValue)
        self.schedular.runner.simulationProgressed.connect(self.kkitRunView.getCentralWidget().changeBgSize)
        self.schedular.runner.simulationReset.connect(self.kkitRunView.getCentralWidget().resetColor)
        self.schedular.runner.simulationReset.connect(self.setSolver)

        return self._centralWidget

    def setSolver(self):
        self.kkitRunView.getCentralWidget().addSolver(self.getSchedulingDockWidget().widget().solver)

    def getCentralWidget(self):
        if self._centralWidget is None:
            self.createCentralWidget()
        return self._centralWidget

class KkitRunView(MooseEditorView):

    #def __init__(self, plugin,dataTable):
    def __init__(self, plugin):
        MooseEditorView.__init__(self, plugin)
        #self.dataTable =dataTable
        self.plugin = plugin
    '''
    def getToolPanes(self):
        return super(KkitRunView, self).getToolPanes()

    def getLibraryPane(self):
        return super(KkitRunView, self).getLibraryPane()

    def getOperationsWidget(self):
        return super(KkitRunView, self).getOperationsPane()

    def getToolBars(self):
        return self._toolBars
    '''
    def getCentralWidget(self):
        if self._centralWidget is None:
            self._centralWidget = kineticRunWidget(self.plugin)
            # self._centralWidget.view.objectSelected.connect(self.plugin.mainWindow.objectEditSlot)
            self._centralWidget.setModelRoot(self.plugin.modelRoot)
        return self._centralWidget

class KkitEditorView(MooseEditorView):
    #def __init__(self, plugin, dataTable):
    def __init__(self, plugin):
        MooseEditorView.__init__(self, plugin)
        ''' EditorView and if kkit model is loaded then save model in XML is allowed '''
        #self.dataTable = dataTable
        self.fileinsertMenu = QtGui.QMenu('&File')
        if not hasattr(self,'SaveModelAction'):
            #self.fileinsertMenu.addSeparator()
            self.saveModelAction = QtGui.QAction('SaveToSBMLFile', self)
            self.saveModelAction.setShortcut(QtGui.QApplication.translate("MainWindow", "Ctrl+S", None, QtGui.QApplication.UnicodeUTF8))
            self.connect(self.saveModelAction, QtCore.SIGNAL('triggered()'), self.SaveModelDialogSlot)
            self.fileinsertMenu.addAction(self.saveModelAction)
        # if not hasattr(self,'SaveModelAction'):
        #     #self.fileinsertMenu.addSeparator()
        #     self.saveModelAction = QtGui.QAction('SaveToGenesisFormat', self)
        #     self.saveModelAction.setShortcut(QtGui.QApplication.translate("MainWindow", "Ctrl+S", None, QtGui.QApplication.UnicodeUTF8))
        #     self.connect(self.saveModelAction, QtCore.SIGNAL('triggered()'), self.SaveToGenesisSlot)
        #     self.fileinsertMenu.addAction(self.saveModelAction)
        self._menus.append(self.fileinsertMenu)

    def SaveModelDialogSlot(self):
        type_sbml = 'SBML'
        filters = {'SBML(*.xml)': type_sbml}
        filename,filter_ = QtGui.QFileDialog.getSaveFileNameAndFilter(None,'Save File','',';;'.join(filters))
        extension = ""
        if str(filename).rfind('.') != -1:
            filename = filename[:str(filename).rfind('.')]
            if str(filter_).rfind('.') != -1:
                extension = filter_[str(filter_).rfind('.'):len(filter_)-1]
        if filename:
            filename = filename+extension
            if filters[str(filter_)] == 'SBML':
                writeerror = moose.writeSBML(str(filename),self.plugin.modelRoot)
                if writeerror:
                    QtGui.QMessageBox.warning(None,'Could not save the Model','\n Error in the consistency check')
                else:
                     QtGui.QMessageBox.information(None,'Saved the Model','\n File Saved to \'{filename}\''.format(filename =filename),QtGui.QMessageBox.Ok)
    '''
    def getToolPanes(self):
        return super(KkitEditorView, self).getToolPanes()

    def getLibraryPane(self):
        return super(KkitEditorView, self).getLibraryPane()

    def getOperationsWidget(self):
        return super(KkitEditorView, self).getOperationsPane()

    def getToolBars(self):
        return self._toolBars
    '''
    def getCentralWidget(self):
        if self._centralWidget is None:
            self._centralWidget = kineticEditorWidget(self.plugin)
            self._centralWidget.setModelRoot(self.plugin.modelRoot)
        return self._centralWidget

class  KineticsWidget(EditorWidgetBase):
    def __init__(self, plugin, *args):
        EditorWidgetBase.__init__(self, *args)
        self.plugin = plugin
        self.border = 5
        self.comptPen = 6
        self.iconScale = 1
        self.arrowsize = 2
        self.defaultComptsize = 5
        self.noPositionInfo = True
        self.reset()
        self.qGraCompt          = {}
        self.mooseId_GObj       = {}
        self.srcdesConnection   = {}
    def reset(self):
        self.createdItem = {}
        #This are created at drawLine
        self.lineItem_dict = {}
        self.object2line = defaultdict(list)
        self.itemignoreZooming = False

        if hasattr(self,'sceneContainer'):
                self.sceneContainer.clear()
        self.sceneContainer = QtGui.QGraphicsScene(self)
        sceneDim = self.sceneContainer.itemsBoundingRect()
        # if (sceneDim.width() == 0 and sceneDim.height() == 0):
        #     self.sceneContainer.setSceneRect(0,0,30,30)
        # else:
        #elf.sceneContainer.setSceneRect(self.sceneContainer.itemsBoundingRect())
        self.sceneContainer.setBackgroundBrush(QColor(230,220,219,120))

    def updateModelView(self):
        self.getMooseObj()
        if not self.m:
            #print "here the empty model "
            #At the time of model building
            # when we want an empty GraphicView while creating new model,
            # then remove all the view and add an empty view
            if hasattr(self, 'view') and isinstance(self.view, QtGui.QWidget):
                self.layout().removeWidget(self.view)
           #self.sceneContainer.setSceneRect(-self.width()/2,-self.height()/2,self.width(),self.height())
            self.view = GraphicalView(self.modelRoot,self.sceneContainer,self.border,self,self.createdItem)
            if isinstance(self,kineticEditorWidget):
                self.view.setRefWidget("editorView")
                self.view.setAcceptDrops(True)
            elif isinstance(self,kineticRunWidget):
                self.view.setRefWidget("runView")
            self.connect(self.view, QtCore.SIGNAL("dropped"), self.objectEditSlot)
            hLayout = QtGui.QGridLayout(self)
            self.setLayout(hLayout)
            hLayout.addWidget(self.view,0,0)
        else:
            self.mooseObjOntoscene()
            self.drawLine_arrow()
            # Already created Model
            # maxmium and minimum coordinates of the objects specified in kkit file.

            if hasattr(self, 'view') and isinstance(self.view, QtGui.QWidget):
                self.layout().removeWidget(self.view)
            self.view = GraphicalView(self.modelRoot,self.sceneContainer,self.border,self,self.createdItem)
            if isinstance(self,kineticEditorWidget):
                self.view.setRefWidget("editorView")
                self.view.setAcceptDrops(True)
                hLayout = QtGui.QGridLayout(self)
                self.setLayout(hLayout)
                hLayout.addWidget(self.view)
            elif isinstance(self,kineticRunWidget):
                self.view.setRefWidget("runView")
                hLayout = QtGui.QGridLayout(self)
                self.setLayout(hLayout)
                hLayout.addWidget(self.view)
                self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
    def getMooseObj(self):
        #This fun call 2 more function
        # -- setupMeshObj(self.modelRoot),
        #    ----self.meshEntry has [meshEnt] = function: {}, Pool: {} etc
        # setupItem
        self.m = wildcardFind(self.modelRoot+'/##[ISA=ChemCompt]')
        if self.m:
            self.xmin = 0.0
            self.xmax = 1.0
            self.ymin = 0.0
            self.ymax = 1.0
            self.autoCordinatepos = {}
            self.srcdesConnection = {}

            #self.meshEntry.clear= {}
            # Compartment and its members are setup
            self.meshEntry,self.xmin,self.xmax,self.ymin,self.ymax,self.noPositionInfo = setupMeshObj(self.modelRoot)
            self.autocoordinates = False
            if self.srcdesConnection:
                self.srcdesConnection.clear()
            else:
                self.srcdesConnection = {}
            setupItem(self.modelRoot,self.srcdesConnection)

            if self.noPositionInfo:
                self.autocoordinates = True

                self.xmin,self.xmax,self.ymin,self.ymax,self.autoCordinatepos = autoCoordinates(self.meshEntry,self.srcdesConnection)
            # TODO: size will be dummy at this point, busizet I need the availiable size from the Gui
            if isinstance(self,kineticEditorWidget):
                # w = self.width()
                # h = self.height()
                # self.size =  QtCore.QSize(w ,h)
                # print("Window Size : ", self.size)
                self.size= QtCore.QSize(1000 ,550)
            else:
                self.size = QtCore.QSize(500 ,550)
            #self.size = QtCore.QSize(300,400)

            # Scale factor to translate the x -y position to fit the Qt graphicalScene, scene width. """
            if self.xmax-self.xmin != 0:
                self.xratio = (self.size.width()-10)/(self.xmax-self.xmin)
            else: self.xratio = self.size.width()-10

            if self.ymax-self.ymin:
                self.yratio = (self.size.height()-10)/(self.ymax-self.ymin)
            else: self.yratio = (self.size.height()-10)
    def sizeHint(self):
        return QtCore.QSize(800,400)

    def updateItemSlot(self, mooseObject):
        #This is overridden by derived classes to connect appropriate
        #slot for updating the display item.
        #In this case if the name is updated from the keyboard both in mooseobj and gui gets updation
        changedItem = ''

        for item in self.sceneContainer.items():
            if isinstance(item,PoolItem):
                if mooseObject.getId() == element(item.mobj).getId():
                    item.updateSlot()
                    #once the text is edited in editor, laydisplay gets updated in turn resize the length, positionChanged signal shd be emitted
                    self.positionChange(mooseObject)

    def updateColorSlot(self,mooseObject, color):
        #Color slot for changing background color for PoolItem from objecteditor
        anninfo = moose.Annotator(mooseObject.path+'/info')
        textcolor,bgcolor = getColor(anninfo)
        anninfo.color = str(color.name())
        item = self.mooseId_GObj[mooseObject]
        if (isinstance(item,PoolItem) or isinstance(item,EnzItem) or isinstance(item,MMEnzItem) ):
            item.updateColor(color)

    def mooseObjOntoscene(self):
        #  All the compartments are put first on to the scene \
        #  Need to do: Check With upi if empty compartments exist
        self.qGraCompt   = {}
        self.mooseId_GObj = {}
        if self.qGraCompt:
            self.qGraCompt.clear()
        else:
            self.qGraCompt = {}
        if self.mooseId_GObj:
            self.mooseId_GObj.clear()
        else:
            self.mooseId_GObj = {}

        for cmpt in sorted(self.meshEntry.iterkeys()):
            self.createCompt(cmpt)
            self.qGraCompt[cmpt]
            #comptRef = self.qGraCompt[cmpt]

        #Enzymes of all the compartments are placed first, \
        #     so that when cplx (which is pool object) queries for its parent, it gets its \
        #     parent enz co-ordinates with respect to QGraphicsscene """

        for cmpt,memb in self.meshEntry.items():
            for enzObj in find_index(memb,'enzyme'):
                enzinfo = enzObj.path+'/info'
                if enzObj.className == 'Enz':
                    enzItem = EnzItem(enzObj,self.qGraCompt[cmpt])
                else:
                    enzItem = MMEnzItem(enzObj,self.qGraCompt[cmpt])
                self.mooseId_GObj[element(enzObj.getId())] = enzItem
                self.setupDisplay(enzinfo,enzItem,"enzyme")

                #self.setupSlot(enzObj,enzItem)
        for cmpt,memb in self.meshEntry.items():
            #print "cmpt ========== memb ",cmpt,memb

            for poolObj in find_index(memb,'pool'):
                poolinfo = poolObj.path+'/info'
                #depending on Editor Widget or Run widget pool will be created a PoolItem or PoolItemCircle
                poolItem = self.makePoolItem(poolObj,self.qGraCompt[cmpt])
                self.mooseId_GObj[element(poolObj.getId())] = poolItem
                self.setupDisplay(poolinfo,poolItem,"pool")

            for reaObj in find_index(memb,'reaction'):
                reainfo = reaObj.path+'/info'
                reaItem = ReacItem(reaObj,self.qGraCompt[cmpt])
                self.setupDisplay(reainfo,reaItem,"reaction")
                self.mooseId_GObj[element(reaObj.getId())] = reaItem

            for tabObj in find_index(memb,'table'):
                tabinfo = tabObj.path+'/info'
                tabItem = TableItem(tabObj,self.qGraCompt[cmpt])
                self.setupDisplay(tabinfo,tabItem,"tab")
                self.mooseId_GObj[element(tabObj.getId())] = tabItem

            for funcObj in find_index(memb,'function'):
                funcinfo = moose.element(funcObj.parent).path+'/info'
                funcParent =self.mooseId_GObj[element(funcObj.parent)]
                funcItem = FuncItem(funcObj,funcParent)
                self.mooseId_GObj[element(funcObj.getId())] = funcItem
                self.setupDisplay(funcinfo,funcItem,"Function")

            for cplxObj in find_index(memb,'cplx'):
                cplxinfo = (cplxObj.parent).path+'/info'
                p = element(cplxObj).parent
                cplxItem = CplxItem(cplxObj,self.mooseId_GObj[element(cplxObj).parent])
                self.mooseId_GObj[element(cplxObj.getId())] = cplxItem
                self.setupDisplay(cplxinfo,cplxItem,"cplx")

        # compartment's rectangle size is calculated depending on children
        for k, v in self.qGraCompt.items():
            rectcompt = v.childrenBoundingRect()
            v.setRect(rectcompt.x()-10,rectcompt.y()-10,(rectcompt.width()+20),(rectcompt.height()+20))
            v.setPen(QtGui.QPen(Qt.QColor(66,66,66,100), self.comptPen, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))

    def createCompt(self,key):
        self.new_Compt = ComptItem(self,0,0,0,0,key)
        self.qGraCompt[key] = self.new_Compt
        self.new_Compt.setRect(10,10,10,10)
        self.sceneContainer.addItem(self.new_Compt)

    def setupDisplay(self,info,graphicalObj,objClass):
        xpos,ypos = self.positioninfo(info)
        # For Reaction and Complex object I have skipped the process to get the facecolor and background color as \
        #    we are not using these colors for displaying the object so just passing dummy color white
        if( objClass == "reaction"  or objClass == "cplx" or objClass == "Function" or objClass == "StimulusTable"):
            textcolor,bgcolor = "white","white"
        else:
            textcolor,bgcolor = getColor(info)
            if bgcolor.name() == "#ffffff" or bgcolor == "white":
                bgcolor = getRandColor()
                Annoinfo = Annotator(info)
                Annoinfo.color = str(bgcolor.name())
        graphicalObj.setDisplayProperties(xpos,ypos,textcolor,bgcolor)

    def positioninfo(self,iteminfo):
        if self.noPositionInfo:

            try:
                # kkit does exist item's/info which up querying for parent.path gives the information of item's parent
                x,y = self.autoCordinatepos[(element(iteminfo).parent).path]
            except:
                # But in Cspace reader doesn't create item's/info, up on querying gives me the error which need to change\
                # in ReadCspace.cpp, at present i am taking care b'cos i don't want to pass just the item where I need to check\
                # type of the object (rea,pool,enz,cplx,tab) which I have already done.
                parent, child = posixpath.split(iteminfo)
                x,y = self.autoCordinatepos[parent]
            ypos = (y-self.ymin)*self.yratio
        else:
            x = float(element(iteminfo).getField('x'))
            y = float(element(iteminfo).getField('y'))
            #Qt origin is at the top-left corner. The x values increase to the right and the y values increase downwards \
            #as compared to Genesis codinates where origin is center and y value is upwards, that is why ypos is negated
            ypos = -(y-self.ymin)*self.yratio
        xpos = (x-self.xmin)*self.xratio

        return(xpos,ypos)

    def drawLine_arrow(self, itemignoreZooming=False):
        for inn,out in self.srcdesConnection.items():
            #print "inn ",inn, " out ",out
            # self.srcdesConnection is dictionary which contains key,value \
            #    key is Enzyme or Reaction  and value [[list of substrate],[list of product]] (tuple)
            #    key is Function and value is [list of pool] (list)

            #src = self.mooseId_GObj[inn]
            if isinstance(out,tuple):
                src = self.mooseId_GObj[inn]
                if len(out[0])== 0:
                    print inn.className + ':' +inn.name+ " doesn't output message"
                else:
                    for items in (items for items in out[0] ):
                        des = self.mooseId_GObj[element(items[0])]
                        self.lineCord(src,des,items,itemignoreZooming)
                if len(out[1]) == 0:
                    print inn.className + ':' +inn.name+ " doesn't output message"
                else:
                    for items in (items for items in out[1] ):
                        des = self.mooseId_GObj[element(items[0])]
                        self.lineCord(src,des,items,itemignoreZooming)
            elif isinstance(out,list):
                if len(out) == 0:
                    print "Func pool doesn't have sumtotal"
                else:
                    src = self.mooseId_GObj[inn]
                    for items in (items for items in out ):
                        des = self.mooseId_GObj[element(items[0])]
                        self.lineCord(src,des,items,itemignoreZooming)
    def lineCord(self,src,des,type_no,itemignoreZooming):
        srcdes_list = []
        endtype = type_no[1]
        line = 0
        if (src == "") and (des == ""):
            print "Source or destination is missing or incorrect"
            return
        srcdes_list = [src,des,endtype,line]
        arrow = calcArrow(srcdes_list,itemignoreZooming,self.iconScale)
        self.drawLine(srcdes_list,arrow)

        while(type_no[2] > 1 and line <= (type_no[2]-1)):
            srcdes_list =[src,des,endtype,line]
            arrow = calcArrow(srcdes_list,itemignoreZooming,self.iconScale)
            self.drawLine(srcdes_list,arrow)
            line = line +1

        if type_no[2] > 5:
            print "Higher order reaction will not be displayed"

    def drawLine(self,srcdes_list,arrow):
        src = srcdes_list[0]
        des = srcdes_list[1]
        endtype = srcdes_list[2]
        line = srcdes_list[3]
        source = element(next((k for k,v in self.mooseId_GObj.items() if v == src), None))
        for l,v,o in self.object2line[src]:
            if v == des and o ==line:
                l.setPolygon(arrow)
                arrowPen = l.pen()
                arrowPenWidth = self.arrowsize*self.iconScale
                arrowPen.setColor(l.pen().color())
                arrowPen.setWidth(arrowPenWidth)
                l.setPen(arrowPen)
                return
        qgLineitem = self.sceneContainer.addPolygon(arrow)
        qgLineitem.setParentItem(src.parentItem())
        pen = QtGui.QPen(QtCore.Qt.green, 0, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin)
        pen.setWidth(self.arrowsize)
        # Green is default color moose.ReacBase and derivatives - already set above
        if  isinstance(source, EnzBase):
            if ( (endtype == 's') or (endtype == 'p')):
                pen.setColor(QtCore.Qt.red)
            elif(endtype != 'cplx'):
                p1 = (next((k for k,v in self.mooseId_GObj.items() if v == src), None))
                pinfo = p1.path+'/info'
                color,bgcolor = getColor(pinfo)
                #color = QColor(color[0],color[1],color[2])
                pen.setColor(color)
        elif isinstance(source, moose.PoolBase) or isinstance(source,moose.Function):
            pen.setColor(QtCore.Qt.blue)
        elif isinstance(source,moose.StimulusTable):
            pen.setColor(QtCore.Qt.yellow)
        self.lineItem_dict[qgLineitem] = srcdes_list
        self.object2line[ src ].append( ( qgLineitem, des,line,) )
        self.object2line[ des ].append( ( qgLineitem, src,line, ) )
        qgLineitem.setPen(pen)

    def positionChange(self,mooseObject):
        #If the item position changes, the corresponding arrow's are calculated
        mobj = self.mooseId_GObj[element(mooseObject)]
        self.updateArrow(mobj)
        for k, v in self.qGraCompt.items():
            rectcompt = v.childrenBoundingRect()
            v.setRect(rectcompt.x()-10,rectcompt.y()-10,(rectcompt.width()+20),(rectcompt.height()+20))

    def updateArrow(self,qGTextitem):
        #if there is no arrow to update then return
        if qGTextitem not in self.object2line:
            return
        listItem = self.object2line[qGTextitem]
        #print "updateArrow ",qGTextitem, listItem
        for ql, va,order in self.object2line[qGTextitem]:
            srcdes = []
            srcdes = self.lineItem_dict[ql]
            # Checking if src (srcdes[0]) or des (srcdes[1]) is ZombieEnz,
            # if yes then need to check if cplx is connected to any mooseObject,
            # so that when Enzyme is moved, cplx connected arrow to other mooseObject(poolItem) should also be updated
            if( type(srcdes[0]) == EnzItem):
                self.cplxUpdatearrow(srcdes[0])
            elif( type(srcdes[1]) == EnzItem):
                self.cplxUpdatearrow(srcdes[1])
            # For calcArrow(src,des,endtype,itemignoreZooming) is to be provided
            arrow = calcArrow(srcdes,self.itemignoreZooming,self.iconScale)
            ql.setPolygon(arrow)

    def cplxUpdatearrow(self,srcdes):
        # srcdes which is 'EnzItem' from this,get ChildItems are retrived (b'cos cplx is child of zombieEnz)
        #And cplxItem is passed for updatearrow
        for item in srcdes.childItems():
            if isinstance(item,CplxItem):
                self.updateArrow(item)

    def deleteSolver(self):
        print "solver deleteSolver"
        if moose.wildcardFind(self.modelRoot+'/##[ISA=ChemCompt]'):
            compt = moose.wildcardFind(self.modelRoot+'/##[ISA=ChemCompt]')
            if ( moose.exists( compt[0].path+'/stoich' ) ):
                moose.delete( compt[0].path+'/stoich' )
                moose.delete( compt[0].path+'/ksolve' )

    def positionChange1(self,mooseObject):
        #If the item position changes, the corresponding arrow's are calculated
        if ( (isinstance(element(mooseObject),CubeMesh)) or (isinstance(element(mooseObject),CylMesh))):
            v = self.qGraCompt[mooseObject]
            for rectChilditem in v.childItems():
                self.updateArrow(rectChilditem)
        else:
            mobj = self.mooseId_GObj[mooseObject.getId()]
            self.updateArrow(mobj)
            mooseObjcompt = self.findparent(mooseObject)
            v = self.qGraCompt[mooseObjcompt]
            childBoundingRect = v.childrenBoundingRect()
            comptBoundingRect = v.boundingRect()
            rectcompt = comptBoundingRect.united(childBoundingRect)
            comptPen = v.pen()
            comptWidth =  5
            comptPen.setWidth(comptWidth)
            v.setPen(comptPen)
            if not comptBoundingRect.contains(childBoundingRect):
                v.setRect(rectcompt.x()-comptWidth,rectcompt.y()-comptWidth,rectcompt.width()+(comptWidth*2),rectcompt.height()+(comptWidth*2))

class kineticEditorWidget(KineticsWidget):
    def __init__(self, plugin,*args):

        KineticsWidget.__init__(self, plugin, *args)
        self.plugin = plugin
        self.insertMenu = QtGui.QMenu('&Insert')
        self._menus.append(self.insertMenu)
        self.insertMapper = QtCore.QSignalMapper(self)
        classlist = ['CubeMesh','CylMesh','Pool','BufPool','Function','Reac','Enz','MMenz','StimulusTable']
        self.toolTipinfo = { "CubeMesh":"",
                             "CylMesh" : "",
                             "Pool":"A Pool is a collection of molecules of a given species in a given cellular compartment.\n It can undergo reactions that convert it into other pool(s). \nParameters: initConc (Initial concentration), diffConst (diffusion constant). Variable: conc (Concentration)",
                             "BufPool":"A BufPool is a buffered pool. \nIt is a collection of molecules of a given species in a given cellular compartment, that are always present at the same concentration.\n This is set by the initConc parameter. \nIt can undergo reactions in the same way as a pool.",
                             "Function":"A Func computes an arbitrary mathematical expression of one or more input concentrations of Pools. The output can be used to control the concentration of another Pool, or as an input to another Func",
                             "StimulusTable":"A StimulusTable stores an array of values that are read out during a simulation, and typically control the concentration of one of the pools in the model. \nParameters: size of table, values of entries, start and stop times, and an optional loopTime that defines the period over which the StimulusTable should loop around to repeat its values",
                             "Reac":"A Reac is a chemical reaction that converts substrates into products, and back. \nThe rates of these conversions are specified by the rate constants Kf and Kb respectively.",
                             "MMenz":"An MMenz is the Michaelis-Menten version of an enzyme activity of a given Pool.\n The MMenz must be created on a pool and can only catalyze a single reaction with a specified substrate(s). \nIf a given enzyme species can have multiple substrates, then multiple MMenz activites must be created on the parent Pool. \nThe rate of an MMenz is V [S].[E].kcat/(Km + [S]). There is no enzyme-substrate complex. Parameters: Km and kcat.",
                             "Enz":"An Enz is an enzyme activity of a given Pool. The Enz must be created on a pool and can only catalyze a single reaction with a specified substrate(s). \nIf a given enzyme species can have multiple substrates, then multiple Enz activities must be created on the parent Pool. \nThe reaction for an Enz is E + S <===> E.S ---> E + P. \nThis means that a new Pool, the enzyme-substrate complex E.S, is always formed when you create an Enz. \nParameters: Km and kcat, or alternatively, K1, K2 and K3. Km = (K2+K3)/K1"

                             }
        insertMapper, actions = self.getInsertActions(classlist)
        for action in actions:
            self.insertMenu.addAction(action)
            doc = self.toolTipinfo[str(action.text())]
            if doc == "":
                classname = str(action.text())
                doc = moose.element('/classes/%s' % (classname)).docs
                doc = doc.split('Description:')[-1].split('Name:')[0].strip()
            action.setToolTip(doc)

    def GrViewresize(self,event):
        #when Gui resize and event is sent which inturn call resizeEvent of qgraphicsview
        pass
        #self.view.resizeEvent1(event)

    def makePoolItem(self, poolObj, qGraCompt):
        return PoolItem(poolObj, qGraCompt)

    def getToolBars(self):
        #Add specific tool items with respect to kkit
        if not hasattr(self, '_insertToolBar'):
            self._insertToolBar = QtGui.QToolBar('Insert')
            self._toolBars.append(self._insertToolBar)
            for action in self.insertMenu.actions():
                button = MToolButton()
                button.setDefaultAction(action)
                #set the unicode instead of image by setting
                #button.setText(unicode(u'\u20de'))
                button.setIcon(QtGui.QIcon("~/../../Docs/images/classIcon/"+action.text()+".png"))
                #button.setIconSize(QtCore.QSize(200,200))
                self._insertToolBar.addWidget(button)
        return self._toolBars

class kineticRunWidget(KineticsWidget):
    def __init__(self, plugin,*args):
        KineticsWidget.__init__(self, plugin,*args)

    def showEvent(self, event):
        self.refresh()
        # pass
    def refresh(self):
        self.sceneContainer.clear()
        self.Comptexist = wildcardFind(self.modelRoot+'/##[ISA=ChemCompt]')
        if self.Comptexist:
            self.getMooseObj()
            self.mooseObjOntoscene()
            self.drawLine_arrow(itemignoreZooming=False)

    def makePoolItem(self, poolObj, qGraCompt):
        return PoolItemCircle(poolObj, qGraCompt)

    def getToolBars(self):
        return self._toolBars

    def updateValue(self):
        for item in self.sceneContainer.items():
            if isinstance(item,ReacItem) or isinstance(item,MMEnzItem) or isinstance(item,EnzItem) or isinstance(item,PoolItemCircle) or isinstance(item,CplxItem):
                item.updateValue(item.mobj)

    def changeBgSize(self):
        for item in self.sceneContainer.items():
            if isinstance(item,PoolItemCircle):
                initialConc = moose.element(item.mobj).concInit
                presentConc = moose.element(item.mobj).conc
                if initialConc != 0:
                    ratio = presentConc/initialConc
                else:
                    # multipying by 1000 b'cos moose concentration is in milli in moose
                    ratio = presentConc
                item.updateRect(math.sqrt(abs(ratio)))

    def resetColor(self):
        for item in self.sceneContainer.items():
            if isinstance(item,PoolItemCircle):
                item.returnEllispeSize()


    '''
    def keyPressEvent(self,event):
        # key1 = event.key() # key event does not distinguish between capital and non-capital letters
        key = event.text().toAscii().toHex()
        if (key ==  '41'): # 'A' fits the view to iconScale factor
            itemignoreZooming = False
            self.updateItemTransformationMode(itemignoreZooming)
            self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
            self.drawLine_arrow(itemignoreZooming=False)

        elif (key == '2e'): # '.' key, lower case for '>' zooms in
            self.view.scale(1.1,1.1)

        elif (key == '2c'): # ',' key, lower case for '<' zooms in
            self.view.scale(1/1.1,1/1.1)

        elif (key == '3c'): # '<' key. zooms-in to iconScale factor
            self.iconScale *= 0.8
            self.updateScale( self.iconScale )

        elif (key == '3e'): # '>' key. zooms-out to iconScale factor
            self.iconScale *= 1.25
            self.updateScale( self.iconScale )

        elif (key == '61'):  # 'a' fits the view to initial value where iconscale=1
            self.iconScale = 1
            self.updateScale( self.iconScale )
            self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)

    def updateScale( self, scale ):
        for item in self.sceneContainer.items():
            if isinstance(item,KineticsDisplayItem):
                item.refresh(scale)
                #iteminfo = item.mobj.path+'/info'
                #xpos,ypos = self.positioninfo(iteminfo)
                xpos = item.scenePos().x()
                ypos = item.scenePos().y()

                if isinstance(item,ReacItem) or isinstance(item,EnzItem) or isinstance(item,MMEnzItem):
                     item.setGeometry(xpos,ypos,
                                     item.gobj.boundingRect().width(),
                                     item.gobj.boundingRect().height())
                elif isinstance(item,CplxItem):
                    item.setGeometry(item.gobj.boundingRect().width()/2,item.gobj.boundingRect().height(),
                                     item.gobj.boundingRect().width(),
                                     item.gobj.boundingRect().height())
                elif isinstance(item,PoolItem):
                     item.setGeometry(xpos, ypos,item.gobj.boundingRect().width()
                                     +PoolItem.fontMetrics.width('  '),
                                     item.gobj.boundingRect().height())
                     item.bg.setRect(0, 0, item.gobj.boundingRect().width()+PoolItem.fontMetrics.width('  '), item.gobj.boundingRect().height())

        self.drawLine_arrow(itemignoreZooming=False)
        for k, v in self.qGraCompt.items():
            rectcompt = v.childrenBoundingRect()
            comptPen = v.pen()
            comptWidth =  self.defaultComptsize*self.iconScale
            comptPen.setWidth(comptWidth)
            v.setPen(comptPen)
            v.setRect(rectcompt.x()-comptWidth,rectcompt.y()-comptWidth,(rectcompt.width()+2*comptWidth),(rectcompt.height()+2*comptWidth))
    '''
    def addSolver(self,solver):
        #print "solver",solver
        compt = moose.wildcardFind(self.modelRoot+'/##[ISA=ChemCompt]')
        comptinfo = moose.Annotator(moose.element(compt[0]).path+'/info')
        previousSolver = comptinfo.solver
        print "solver from kkit ",previousSolver
        currentSolver = "GSL"
        if solver == "Gillespie":
            currentSolver = "gssa"
        elif solver == "Runge Kutta":
            currentSolver = "rk"

        if previousSolver != currentSolver:
            if not ( moose.exists( compt[0].path+'/stoich' ) ):
                #previos solver is not the same and current solver and
                # if solver is not set
                #print "condition 1"
                self.setCompartmentSolver(currentSolver)

            if ( moose.exists( compt[0].path+'/stoich' ) ):
                #previos solver is not the same and current solver and
                # if solver is set
                #print " condition 2",previousSolver,currentSolver
                self.deleteSolver()
                self.setCompartmentSolver(compt,currentSolver)
            comptinfo.solver = currentSolver

    def setCompartmentSolver(self,compt,solver):
            #print "solver in setSolver",solver
            if ( solver == 'gsl' ):
                ksolve = moose.Ksolve( compt[0].path+'/ksolve' )
            if ( solver == 'gssa' ):
                ksolve = moose.Gsolve( compt[0].path+'/ksolve' )
            if ( solver != 'ee' ):
                stoich = moose.Stoich( compt[0].path+'/stoich' )
                ksolve = moose.Ksolve( compt[0].path+'/ksolve' )
                stoich.compartment = moose.element(compt[0])
                stoich.ksolve = ksolve
                stoich.path = compt[0].path+"/##"



if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    size = QtCore.QSize(1024 ,768)
    #modelPath = 'Kholodenko'
    modelPath = 'acc27'
    #modelPath = 'acc8'
    #modelPath = '3ARECB'
    #modelPath = '3AreacB'
    #modelPath = '5AreacB'
    itemignoreZooming = False
    try:
        filepath = '../../Demos/Genesis_files/'+modelPath+'.g'
        filepath = '/home/harsha/genesis_files/gfile/'+modelPath+'.g'
        print filepath
        f = open(filepath, "r")
        loadModel(filepath,'/'+modelPath)

        #moose.le('/'+modelPath+'/kinetics')
        dt = KineticsWidget()
        dt.modelRoot ='/'+modelPath
        ''' Loading moose signalling model in python '''
        #execfile('/home/harsha/BuildQ/Demos/Genesis_files/scriptKineticModel.py')
        #dt.modelRoot = '/model'

        dt.updateModelView()
        dt.show()

    except  IOError, what:
      (errno, strerror) = what
      print "Error number",errno,"(%s)" %strerror
      sys.exit(0)
    sys.exit(app.exec_())
