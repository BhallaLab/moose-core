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
from kkitCalcArrow import *
from PyQt4 import QtGui, QtCore, Qt
import math

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
            self.yratio = (self.size.height()-10)/(self.ymax-self.ymin)
        else: self.yratio = (self.size.height()-10)

        #A map b/w moose compartment key with QGraphicsObject
        self.qGraCompt = {}
        
        #A map between mooseId of all the mooseObject (except compartment) with QGraphicsObject
        self.mooseId_GObj = {}
        self.srcdesConnection = {}
        self.border = 5
        self.arrowsize = 2
        self.iconScale = 1
        self.defaultComptsize = 5
        self.itemignoreZooming = False


        self.lineItem_dict = {}
        
        self.object2line = defaultdict(list)
        hLayout = QtGui.QGridLayout(self)
        self.setLayout(hLayout)
        self.sceneContainer = QtGui.QGraphicsScene(self)
        self.sceneContainer.setSceneRect(self.sceneContainer.itemsBoundingRect())
        self.sceneContainer.setBackgroundBrush(QtGui.QColor(230,220,219,120))

        
        """ Compartment and its members are put on the qgraphicsscene """
        self.mooseObjOntoscene()
        self.setupItem(self.modelRoot,self.srcdesConnection)
        self.drawLine_arrow(itemignoreZooming=False)

        self.view = GraphicalView(self.sceneContainer,self.border,self)
        hLayout.addWidget(self.view)
        #self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)

    def drawLine_arrow(self, itemignoreZooming=False):
        for inn,out in self.srcdesConnection.items():
            ''' self.srcdesConnection is dictionary which contains key,value \
                key is Enzyme or Reaction  and value [[list of substrate],[list of product]] (tuple)
                key is FuncBase and value is [list of pool] (list)
            '''
            src = self.mooseId_GObj[inn]
            if isinstance(out,tuple):
                if len(out[0])== 0:
                    print "Reaction or Enzyme doesn't input mssg"
                else:
                    for items in (items for items in out[0] ):
                        des = self.mooseId_GObj[element(items[0]).getId()]
                        self.lineCord(src,des,items[1],itemignoreZooming)
                if len(out[1]) == 0:
                    print "Reaction or Enzyme doesn't output mssg"
                else:
                    for items in (items for items in out[1] ):
                        des = self.mooseId_GObj[element(items[0]).getId()]
                        self.lineCord(src,des,items[1],itemignoreZooming)

            elif isinstance(out,list):
                if len(out) == 0:
                    print "Func pool doesn't have sumtotal"
                else:
                    for items in (items for items in out ):
                        des = self.mooseId_GObj[element(items[0]).getId()]
                        self.lineCord(src,des,items[1],itemignoreZooming)
    
    def lineCord(self,src,des,endtype,itemignoreZooming):
        source = element(next((k for k,v in self.mooseId_GObj.items() if v == src), None))
        desc = element(next((k for k,v in self.mooseId_GObj.items() if v == des), None))
        line = 0
        if (src == "") and (des == ""):
            print "Source or destination is missing or incorrect"
            return 
        srcdes_list = [src,des,endtype]
        arrow = calcArrow(src,des,endtype,itemignoreZooming,self.iconScale)
        for l,v in self.object2line[src]:
            if v == des:
                l.setPolygon(arrow)
                arrowPen = l.pen()
                arrowPenWidth = self.arrowsize*self.iconScale
                arrowPen.setColor(l.pen().color())
                arrowPen.setWidth(arrowPenWidth)
                l.setPen(arrowPen)
                return
        qgLineitem = self.sceneContainer.addPolygon(arrow)
        pen = QtGui.QPen(QtCore.Qt.green, 0, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin)
        pen.setWidth(self.arrowsize)
        #pen.setCosmetic(True)
        # Green is default color moose.ReacBase and derivatives - already set above
        if  isinstance(source, EnzBase):
            if ( (endtype == 's') or (endtype == 'p')):
                pen.setColor(QtCore.Qt.red)
            elif(endtype != 'cplx'):
                p1 = (next((k for k,v in self.mooseId_GObj.items() if v == src), None))
                pinfo = p1.path+'/info'
                color,bgcolor = getColor(pinfo,self.colorMap)
                pen.setColor(color)
        elif isinstance(source, moose.PoolBase):
            pen.setColor(QtCore.Qt.blue)
        self.lineItem_dict[qgLineitem] = srcdes_list
        self.object2line[ src ].append( ( qgLineitem, des) )
        self.object2line[ des ].append( ( qgLineitem, src ) )
        qgLineitem.setPen(pen)


    def setupItem(self,modlePath,cntDict):
        '''This function collects information of what is connected to what. \
           eg. substrate and product connectivity to reaction's and enzyme's \
               sumtotal connectivity to its pool are collected '''

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
            else:
                #ZombieSumFunc adding inputs
                for items in wildcardFind(path):
                    inputlist = []
                    outputlist = []
                    funplist = []
                    nfunplist = []
                    for inpt in items[0].getNeighbors('input'):
                        inputlist.append((inpt,'st'))
                    for funcbase in items[0].getNeighbors('output'): 
                        funplist.append(funcbase)
                    if(len(funplist) > 1): print "SumFunPool has multiple Funpool"
                    else:  cntDict[funplist[0]] = inputlist


    def mooseObjOntoscene(self):
        """  All the compartments are put first on to the scene \
             Need to do: Check With upi if empty compartments exist """
        for cmpt in sorted(self.meshEntry.iterkeys()):
            self.createCompt(cmpt)
            comptRef = self.qGraCompt[cmpt]
        
        """ Enzymes of all the compartments are placed first, \
             so that when cplx (which is pool object) queries for its parent, it gets its \
             parent enz co-ordinates with respect to QGraphicsscene """
        
        for cmpt,memb in self.meshEntry.items():
            for enzObj in self.find_index(memb,'enzyme'):
                enzinfo = enzObj.path+'/info'
                if enzObj.class_ == 'ZEnz':
                    enzItem = EnzItem(enzObj,self.qGraCompt[cmpt])
                else:
                    enzItem = MMEnzItem(enzObj,self.qGraCompt[cmpt])
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
            v.cmptEmitter.connect(v.cmptEmitter,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.positionChange)
            v.cmptEmitter.connect(v.cmptEmitter,QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),self.emitItemtoEditor)

    def setupSlot(self,mooseObj,qgraphicItem):
        self.mooseId_GObj[element(mooseObj).getId()] = qgraphicItem
        qgraphicItem.connect(qgraphicItem,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.positionChange)
        qgraphicItem.connect(qgraphicItem,QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),self.emitItemtoEditor)

    def find_key(self, dic, val):
        """return the key of dictionary dic given the value"""
        return [k for k, v in dic.iteritems() if v == val]

    def positionChange(self,mooseObject):
        #If the item position changes, the corresponding arrow's are calculated
        if isinstance(element(mooseObject),CubeMesh):
            for k, v in self.qGraCompt.items():
                mesh = mooseObject.path+'/mesh[0]'
                if k.path == mesh:
                    for rectChilditem in v.childItems():
                        self.updateArrow(rectChilditem)
        else:
            mobj = self.mooseId_GObj[mooseObject.getId()]
            self.updateArrow(mobj)
            for k, v in self.qGraCompt.items():
                rectcompt = v.childrenBoundingRect()
                v.setRect(rectcompt.x()-10,rectcompt.y()-10,(rectcompt.width()+20),(rectcompt.height()+20))
    
    def updateArrow(self,qGTextitem):
        #if there is no arrow to update then return
        if qGTextitem not in self.object2line:
            return
        listItem = self.object2line[qGTextitem]
        for ql, va in self.object2line[qGTextitem]:
            srcdes = self.lineItem_dict[ql]
            # Checking if src (srcdes[0]) or des (srcdes[1]) is ZombieEnz,
            # if yes then need to check if cplx is connected to any mooseObject, 
            # so that when Enzyme is moved, cplx connected arrow to other mooseObject(poolItem) should also be updated
            if( type(srcdes[0]) == EnzItem):
                self.cplxUpdatearrow(srcdes[0])
            elif( type(srcdes[1]) == EnzItem):
                self.cplxUpdatearrow(srcdes[1])
            
            # For calcArrow(src,des,endtype,itemignoreZooming) is to be provided
            arrow = calcArrow(srcdes[0],srcdes[1],srcdes[2],self.itemignoreZooming,self.iconScale)
            ql.setPolygon(arrow)
    
    def cplxUpdatearrow(self,srcdes):
        ''' srcdes which is 'EnzItem' from this,get ChildItems are retrived (b'cos cplx is child of zombieEnz)
            And cplxItem is passed for updatearrow
        '''
        #Note: Here at this point enzItem has just one child which is cplxItem and childItems returns, PyQt4.QtGui.QGraphicsEllipseItem,CplxItem
        #Assuming CplxItem is always[1], but still check if not[0], if something changes in structure one need to keep an eye.
        if (srcdes.childItems()[1],CplxItem):
            self.updateArrow(srcdes.childItems()[1])
        else:
            self.updateArrow(srcdes.childItems()[0])

    def emitItemtoEditor(self,mooseObject):
        self.emit(QtCore.SIGNAL("itemPressed(PyQt_PyObject)"),mooseObject)

    def setupDisplay(self,info,graphicalObj,objClass):
        xpos,ypos = self.positioninfo(info)
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
            
    def keyPressEvent(self,event):
        # key1 = event.key() # key event does not distinguish between capital and non-capital letters
        key = event.text().toAscii().toHex()
        if key ==  '41': # 'A' fits the view to iconScale factor
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
                xpos = item.scenePos().x()
                ypos = item.scenePos().y()
                item.refresh(scale)
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
            
    def positioninfo(self,iteminfo):
        x =  float(element(iteminfo).getField('x'))
        y = float(element(iteminfo).getField('y'))
        xpos = (x-self.xmin)*self.xratio
        ypos = -(y-self.ymin)*self.yratio
        return(xpos,ypos)
