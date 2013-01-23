import sys
import os
import numpy as np
import config
import pickle 
sys.path.append('plugins')
from default import *
from moose import *
from mplugin import *
from kkitUtil import *
from kkitQGraphics import PoolItem, ReacItem,EnzItem,CplxItem,ComptItem
from kkitViewcontrol import *
from PyQt4 import QtGui, QtCore, Qt

class KkitPlugin(MoosePlugin):
    """Default plugin for MOOSE GUI"""
    def __init__(self, *args):
        #print args
        MoosePlugin.__init__(self, *args)

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
            self.editorView = KkitEditorView(self)
            self.currentView = self.editorView
        return self.editorView


class KkitEditorView(MooseEditorView):
    """Default editor.

    TODO: Implementation - display moose element tree as a tree or as
    boxes inside boxes

    """
    def __init__(self, plugin):
        MooseEditorView.__init__(self, plugin)

    def getToolPanes(self):
        return super(KkitEditorView, self).getToolPanes()

    def getLibraryPane(self):
        return super(KkitEditorView, self).getLibraryPane()

    def getOperationsWidget(self):
        return super(KkitEditorView, self).getOperationsPane()

    def getCentralWidget(self):
        if self._centralWidget is None:
            #self._centralWidget = EditorWidgetBase()
            self._centralWidget = KineticsWidget()
            #print "getCentrelWidget",self.plugin.modelRoot
            self._centralWidget.setModelRoot(self.plugin.modelRoot)
        return self._centralWidget

class  KineticsWidget(DefaultEditorWidget):
    def __init__(self, *args):
        #QtGui.QWidget.__init__(self,parent)
	DefaultEditorWidget.__init__(self, *args)

        #print "KKIT plugin",self.modelRoot
    def Checkthisfun(self):
        print "Check if I can call this function"
    def updateModelView(self):
        """ maxmium and minimum coordinates of the objects specified in kkit file. """
        self.xmin = 0.0
        self.xmax = 1.0
        self.ymin = 0.0
        self.ymax = 1.0
        self.size = QtCore.QSize(1024 ,768)

        """ pickled the color map file """
        colormap_file = open(os.path.join(config.settings[config.KEY_COLORMAP_DIR], 'rainbow2.pkl'),'rb')
        self.colorMap = pickle.load(colormap_file)
        colormap_file.close()
        
        self.setupMeshObj()
        #for key,value in self.meshEntry.items():             print key,value
            
        if self.noPositionInfo:
            QtGui.QMessageBox.warning(self, 
                                      'No coordinates found', 
                                      'Kinetic layout works only \
                                       for models using kkit8 or later')
            raise Exception('Unsupported kkit version')
        
        """Scale factor to translate the x -y position from kkit to Qt coordinates. \
           Qt origin is at the top-left corner. The x values increase to the right and the y values increase downwards \
             as compared to Genesis codinates y value is upwards """

        if self.xmax-self.xmin != 0:
            self.xratio = (self.size.width()-10)/(self.xmax-self.xmin)
        else: self.xratio = self.size.width()-10


        if self.ymax-self.ymin:
            self.yratio =- (self.size.height()-10)/(self.ymax-self.ymin)
        else: self.yratio =- (self.size.height()-10)

        #A map b/w moose compartment key with QGraphicsObject
        self.qGraCompt = {}
        
        #A map between mooseId of all the mooseObject (except compartment) with QGraphicsObject
        self.mooseId_GObj = {}
        self.srcdesConnection = {}
        self.border = 5

        hLayout = QtGui.QGridLayout(self)
        self.setLayout(hLayout)
        self.sceneContainer = QtGui.QGraphicsScene(self)
        self.sceneContainer.setSceneRect(self.sceneContainer.itemsBoundingRect())
        self.sceneContainer.setBackgroundBrush(QtGui.QColor(230,220,219,120))

        
        """ Compartment and its members are put on the qgraphicsscene """
        self.mooseObjOntoscene()
        print "pat",self.modelRoot
        self.setupItem(self.modelRoot,self.srcdesConnection)
        self.view = GraphicalView(self.sceneContainer,self.border,self)
        hLayout.addWidget(self.view)
        #self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
    
    def setupItem(self,modlePath,cntDict):
        ''' Reaction's and enzyme's substrate and product and sumtotal is collected '''
        zombieType = ['ReacBase','EnzBase','FuncBase']
        for baseObj in zombieType:
            path = modlePath+'/##[ISA='+baseObj+']'
            if baseObj != 'FuncBase':
                for items in wildcardFind(path):
                    sublist = []
                    prdlist = []
                    for sub in items[0].getNeighbors('sub'): 
                        sublist.append((sub,'s'))
                    for prd in items[0].getNeighbors('prd'):
                        prdlist.append((prd,'p'))
                    if (baseObj == 'CplxEnzBase') :
                        for enzpar in items[0].getNeighbors('toEnz'):
                            sublist.append((enzpar,'t'))
                        for cplx in items[0].getNeighbors('cplxDest'):
                            prdlist.append((cplx,'cplx'))
                    if (baseObj == 'EnzBase'):
                        for enzpar in items[0].getNeighbors('enzDest'):
                            sublist.append((enzpar,'t'))
                    cntDict[items] = sublist,prdlist
                    #print "s and p",sublist,prdlist
            else:
                #ZombieSumFunc adding inputs
                print "path",path,wildcardFind(path)
                for items in wildcardFind(path):
                    inputlist = []
                    outputlist = []
                    funplist = []
                    nfunplist = []
                    print "!",items[0].getNeighbors('input')
                    for inpt in items[0].getNeighbors('input'):
                        inputlist.append((inpt,'st'))
                    print "inputlist",inputlist
                    for zfun in items[0].getNeighbors('output'): funplist.append(zfun)
                    print "f",funplist
                    for i in funplist: nfunplist.append(element(i).getId())
                    print 'n',nfunplist
                    nfunplist = list(set(nfunplist))
                    print 'n1',nfunplist
                    if(len(nfunplist) > 1): print "SumFunPool has multiple Funpool"
                    else:
                        for el in funplist:
                            if(element(el).getId() == nfunplist[0]):
                                cntDict[element(el)] = inputlist
                                return

    def mooseObjOntoscene(self):
        """  All the compartments are put first on to the scene \
             Need to do: Check With upi if empty compartments exist """
        for cmpt in sorted(self.meshEntry.iterkeys()):
            print "cmpt",cmpt
            self.createCompt(cmpt)
            comptRef = self.qGraCompt[cmpt]
        
        """ Enzymes of all the compartments are placed first, \
             so that when cplx (which is pool object) queries for its parent, it gets its \
             parent enz co-ordinates with respect to QGraphicsscene """
        
        for cmpt,memb in self.meshEntry.items():
            for enzObj in self.find_index(memb,'enzyme'):
                enzinfo = enzObj.path+'/info'
                enzItem = EnzItem(enzObj,self.qGraCompt[cmpt])
                self.setupDisplay(enzinfo,enzItem,"enzyme")
                self.setupSlot(enzObj,enzItem)

        for cmpt,memb in self.meshEntry.items():
            for poolObj in self.find_index(memb,'pool'):
                poolinfo = poolObj.path+'/info'
                poolItem = PoolItem(poolObj,self.qGraCompt[cmpt])
                self.setupDisplay(poolinfo,poolItem,"pool")
                self.setupSlot(poolObj,poolItem)

            for cplxObj in self.find_index(memb,'cplx'):
                cplxinfo = (cplxObj[0].parent).path+'/info'
                cplxItem = CplxItem(cplxObj,self.mooseId_GObj[element(cplxObj[0]).parent.getId()])
                self.setupDisplay(cplxinfo,cplxItem,"cplx")
                self.setupSlot(cplxObj,cplxItem)

            for reaObj in self.find_index(memb,'reaction'):
                reainfo = reaObj.path+'/info'
                reaItem = ReacItem(reaObj,self.qGraCompt[cmpt])
                self.setupDisplay(reainfo,reaItem,"reaction")
                self.setupSlot(reaObj,reaItem)


        ''' compartment's rectangle size is calculated depending on children '''
        for k, v in self.qGraCompt.items():
            rectcompt = v.childrenBoundingRect()
            v.setRect(rectcompt.x()-10,rectcompt.y()-10,(rectcompt.width()+20),(rectcompt.height()+20))
            v.setPen(QtGui.QPen(Qt.QColor(66,66,66,100), 5, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))

    def setupSlot(self,mooseObj,qgraphicItem):
        self.mooseId_GObj[element(mooseObj).getId()] = qgraphicItem
        qgraphicItem.connect(qgraphicItem,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.positionChange)
        qgraphicItem.connect(qgraphicItem,QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),self.emitItemtoEditor)

    def find_key(self, dic, val):
        """return the key of dictionary dic given the value"""
        return [k for k, v in dic.iteritems() if v == val]

    def positionChange(self,mooseObject):
        #If the item position changes, the corresponding arrow's are calculated
        print "position Changed"
        if isinstance(element(mooseObject),CubeMesh):
            for k, v in self.qGraCompt.items():
                mesh = mooseObject.path+'/mesh[0]'
                if k.path == mesh:
                    for rectChilditem in v.childItems():
                        #self.updateArrow(rectChilditem)
                        pass
        else:
            mobj = self.mooseId_GObj[mooseObject.getId()]
            #self.updateArrow(pool)
            for k, v in self.qGraCompt.items():
                rectcompt = v.childrenBoundingRect()
                v.setRect(rectcompt.x()-10,rectcompt.y()-10,(rectcompt.width()+20),(rectcompt.height()+20))

    def emitItemtoEditor(self,mooseObject):
        print "selected"
        self.emit(QtCore.SIGNAL("itemPressed(PyQt_PyObject)"),mooseObject)

    def setupDisplay(self,info,graphicalObj,objClass):
        x = float(element(info).getField('x'))
        y = float(element(info).getField('y'))
        xpos = x*self.xratio
        ypos = y*self.yratio
        """ For Reaction and Complex object I have skipped the process to get the facecolor and background color as \
            we are not using these colors for displaying the object so just passing dummy color white """
        if( (objClass == "reaction" ) or (objClass == "cplx")):
            textcolor,bgcolor = "white","white"
        else:
            textcolor,bgcolor = getColor(info,self.colorMap)

        graphicalObj.setDisplayProperties(xpos,ypos,textcolor,bgcolor)

    
    def createCompt(self,key):
        self.new_Compt = ComptItem(self,0,0,0,0,key)
        self.qGraCompt[key] = self.new_Compt
        self.new_Compt.setRect(10,10,10,10)
        self.sceneContainer.addItem(self.new_Compt)
        
    def find_index(self,value, key):
        """ Value.get(key) to avoid expection which would raise if empty value in dictionary for a given key """
        if value.get(key) != None:
            return value.get(key)
        else:
            raise ValueError('no dict with the key found')

    def setupMeshObj(self):
        ''' Setup compartment and its members pool,reaction,enz cplx under self.meshEntry dictionaries \ 
            self.meshEntry with "key" as compartment, 
            value is key2:list where key2 represents moose object type,list of objects of a perticular type
            e.g self.meshEntry[meshEnt] = { 'reaction': reaction_list,'enzyme':enzyme_list,'pool':poollist,'cplx': cplxlist }
        '''
        self.meshEntry = {}
        xcord = []
        ycord = []
        meshEntryWildcard = self.modelRoot+'/##[TYPE=MeshEntry]'
        for meshEnt in wildcardFind(meshEntryWildcard):
            mollist = []
            realist = []
            enzlist = []
            cplxlist = []
            for reItem in Neutral(meshEnt).getNeighbors('remeshReacs'):
                if isinstance(element(reItem),moose.EnzBase):
                    enzlist.append(reItem)
                else:
                    realist.append(reItem)
                objInfo = reItem.path+'/info'
                xcord.append(float(element(objInfo).getField('x')))
                ycord.append(float(element(objInfo).getField('y')))
                
            for mitem in Neutral(meshEnt).getNeighbors('remesh'):
                """ getNeighbors(remesh) has eliminating GSLStoich """
                if isinstance(element(mitem[0].parent),CplxEnzBase):
                    cplxlist.append(mitem)
                    objInfo = (mitem[0].parent).path+'/info'

                elif isinstance(element(mitem),moose.PoolBase):
                    mollist.append(mitem)
                    objInfo = mitem.path+'/info'
                xcord.append(float(element(objInfo).getField('x')))
                ycord.append(float(element(objInfo).getField('y')))

            self.meshEntry[meshEnt] = {'enzyme':enzlist,
                                       'reaction':realist,
                                       'pool':mollist,
                                       'cplx':cplxlist}
            self.xmin = min(xcord)
            self.xmax = max(xcord)
            self.ymin = min(ycord)
            self.ymax = max(ycord)

            self.noPositionInfo = len(np.nonzero(xcord)[0]) == 0 \
                                  and len(np.nonzero(ycord)[0]) == 0
            
