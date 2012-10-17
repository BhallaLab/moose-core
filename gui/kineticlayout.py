import sys
import  os
from PyQt4 import QtGui,QtCore,Qt
import pickle
import random
import config
import re
import math
from collections import defaultdict
sys.path.append('../python')

from moose import *
global itemignoreZooming
itemignoreZooming = False

class GraphicalView(QtGui.QGraphicsView):
    def __init__(self,parent,border,layoutPt):
        QtGui.QGraphicsView.__init__(self,parent)
        self.setScene(parent)
        self.sceneContainerPt = parent
        self.setDragMode(QtGui.QGraphicsView.RubberBandDrag)
        self.itemSelected = False
        self.customrubberBand=0
        self.rubberbandWidth = 0
        self.rubberbandHeight = 0
        self.moved = False
        self.showpopupmenu = False
        self.border = 6
        self.setRenderHints(QtGui.QPainter.Antialiasing)
        self.layoutPt = layoutPt        
    
    def resizeEvent1(self, event):
        """ zoom when resize! """
        #self.fitInView(self.sceneContainerPt.sceneRect(), Qt.Qt.IgnoreAspectRatio)
        self.fitInView(self.sceneContainerPt.itemsBoundingRect().x()-10,self.sceneContainerPt.itemsBoundingRect().y()-10,self.sceneContainerPt.itemsBoundingRect().width()+20,self.sceneContainerPt.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
        QtGui.QGraphicsView.resizeEvent(self, event)
        
    def wheelEvent(self,event):
        factor = 1.41 ** (event.delta() / 240.0)
        self.scale(factor, factor)
    
    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            self.startingPos = event.pos()
            self.startScenepos = self.mapToScene(self.startingPos)
            self.deviceTransform = self.viewportTransform()
            if config.QT_MINOR_VERSION >= 6:
                # deviceTransform needs to be provided if the scene contains items that ignore transformations,which was introduced in 4.6
                sceneitems = self.sceneContainerPt.itemAt(self.startScenepos,self.deviceTransform)
            else:
                #for below  Qt4.6 there is no view transform for itemAt 
                #and if view is zoom out below 50%  and if textitem object is moved, zooming also happens.
                sceneitems = self.sceneContainerPt.itemAt(self.startScenepos)
            #checking if mouse press position is on any item (in my case textitem or rectcompartment) if none, 
            if ( sceneitems == None):
                QtGui.QGraphicsView.mousePressEvent(self, event)
                #Since qgraphicsrectitem is a item in qt, if I select inside the rectangle it would select the entire
                #rectangle and would not allow me to select the items inside the rectangle so breaking the code by not
                #calling parent class to inherit functionality rather writing custom code for rubberband effect here
            else:
                if( (isinstance(sceneitems.parentWidget(), PoolItem)) or (isinstance(sceneitems.parentWidget(), CplxItem)) or (isinstance(sceneitems, ReacItem)) or (isinstance(sceneitems.parentWidget(),EnzItem)) ):
                    QtGui.QGraphicsView.mousePressEvent(self, event)
                    self.itemSelected = True

                elif(isinstance(sceneitems, ComptItem)):
                    for previousSelection in self.sceneContainerPt.selectedItems():
                        if previousSelection.isSelected() == True:
                            previousSelection.setSelected(0)
                    #Checking if its on the border or inside
                    xs = sceneitems.mapToScene(sceneitems.boundingRect().topLeft()).x()+self.border/2
                    ys = sceneitems.mapToScene(sceneitems.boundingRect().topLeft()).y()+self.border/2
                    xe = sceneitems.mapToScene(sceneitems.boundingRect().bottomRight()).x()-self.border/2
                    ye = sceneitems.mapToScene(sceneitems.boundingRect().bottomRight()).y()-self.border/2
                    xp = self.startScenepos.x()
                    yp = self.startScenepos.y()
                    
                    #if on border rubberband is not started, but called parent class for default implementation
                    if(((xp > xs-self.border/2) and (xp < xs+self.border/2) and (yp > ys-self.border/2) and (yp < ye+self.border/2) )or ((xp > xs+self.border/2) and (xp < xe-self.border/2) and (yp > ye-self.border/2) and (yp < ye+self.border/2) ) or ((xp > xs+self.border/2) and (xp < xe-self.border/2) and (yp > ys-self.border/2) and (yp < ys+self.border/2) ) or ((xp > xe-self.border/2) and (xp < xe+self.border/2) and (yp > ys-self.border/2) and (yp < ye+self.border/2) ) ):
                        QtGui.QGraphicsView.mousePressEvent(self, event)
                        self.itemSelected = True
                        if sceneitems.isSelected() == False:
                            sceneitems.setSelected(1)
                        #if its inside the qgraphicsrectitem then custom code for starting rubberband selection 
                    else:
                        self.customrubberBand = QtGui.QRubberBand(QtGui.QRubberBand.Rectangle,self)
                        self.customrubberBand.setGeometry(QtCore.QRect(self.startingPos,QtCore.QSize()))
                        self.customrubberBand.show()
    
    def mouseMoveEvent(self,event):
        QtGui.QGraphicsView.mouseMoveEvent(self, event)
        if( (self.customrubberBand) and (event.buttons() == QtCore.Qt.LeftButton)):
            self.endingPos = event.pos()
            self.endingScenepos = self.mapToScene(self.endingPos)
            self.rubberbandWidth = self.endingScenepos.x()-self.startScenepos.x()
            self.rubberbandHeight = self.endingScenepos.y()-self.startScenepos.y()
            self.customrubberBand.setGeometry(QtCore.QRect(self.startingPos, event.pos()).normalized())
            #unselecting any previosly selected item in scene
            for preSelectItem in self.sceneContainerPt.selectedItems():
                preSelectItem.setSelected(0)
            #since it custom rubberband I am checking if with in the selected area any textitem, if s then setselected to true
            for items in self.sceneContainerPt.items(self.startScenepos.x(),self.startScenepos.y(),self.rubberbandWidth,self.rubberbandHeight,Qt.Qt.IntersectsItemShape):
                if(isinstance(items.parentWidget(),PoolItem) or isinstance(items.parentWidget(), CplxItem) or isinstance(items, ReacItem) or isinstance(items.parentWidget(),EnzItem)):
                    if(not isinstance(items,ReacItem)):
                        if items.parentWidget().isSelected() == False:
                            items.parentWidget().setSelected(1)
                    else:
                         if items.isSelected() == False:
                            items.setSelected(1)
                        
                        
    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.Qt.ArrowCursor)
        QtGui.QGraphicsView.mouseReleaseEvent(self, event)
        if(self.customrubberBand):
            self.customrubberBand.hide()
            self.customrubberBand = 0
        if event.button() == QtCore.Qt.LeftButton and self.itemSelected == False :
            self.endingPos = event.pos()
            self.endScenepos = self.mapToScene(self.endingPos)
            self.rubberbandWidth = (self.endScenepos.x()-self.startScenepos.x())
            self.rubberbandHeight = (self.endScenepos.y()-self.startScenepos.y())
            selecteditems = self.sceneContainerPt.selectedItems()
            if self.rubberbandWidth != 0 and self.rubberbandHeight != 0 and len(selecteditems) != 0 :
                #self.zoomItem()
                self.showpopupmenu = True
        self.itemSelected = False
        if self.showpopupmenu:
            #Check if entire rect is selected then also it shd work
            popupmenu = QtGui.QMenu('PopupMenu', self)
            self.zoom = QtGui.QAction(self.tr('zoom'), self)
            self.connect(self.zoom, QtCore.SIGNAL('triggered()'), self.zoomItem)
            self.move = QtGui.QAction(self.tr('move'), self)
            self.connect(self.move, QtCore.SIGNAL('triggered()'), self.moveItem)
            #~ popupmenu.addAction(self.delete)
            popupmenu.addAction(self.zoom)
            popupmenu.addAction(self.move)
            popupmenu.exec_(event.globalPos())
        self.showpopupmenu = False

    
    def moveItem(self):
      self.setCursor(Qt.Qt.CrossCursor)

    def zoomItem(self):
        # First items are ignoretransformation,view is fitted and recalculated the arrow
        #global itemignoreZooming = True
        global itemignoreZooming
        itemignoreZooming = True
        self.layoutPt.updateItemTransformationMode()
        self.fitInView(self.startScenepos.x(),self.startScenepos.y(),self.rubberbandWidth,self.rubberbandHeight,Qt.Qt.IgnoreAspectRatio)
        self.layoutPt.drawLine_arrow()
        self.rubberBandactive = False

class KineticsDisplayItem(QtGui.QGraphicsWidget):
    """Base class for display elemenets in kinetics layout"""
    def __init__(self, mooseObject, parent=None):
        QtGui.QGraphicsObject.__init__(self, parent)
        self.mobj = mooseObject
        self.gobj = None

    def setDisplayProperties(self, dinfo):
        self.setGeometry(dinfo.x, dinfo.y)        
    def itemChange(self,change,value):
        if change == QtGui.QGraphicsItem.ItemPositionChange:
            self.emit(QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),element(self.mobj))
        if change == QtGui.QGraphicsItem.ItemSelectedChange and value == True:
            self.emit(QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),element(self.mobj))
        return QtGui.QGraphicsItem.itemChange(self,change,value)


class PoolItem(KineticsDisplayItem):
    """Class for displaying pools. Uses a QGraphicsSimpleTextItem to
    display the name."""    
    #font = QtGui.QFont('times',180)
    #fontMetrics = QtGui.QFontMetrics(font)
    fontMetrics = None
    def __init__(self, *args, **kwargs):
        
        KineticsDisplayItem.__init__(self, *args, **kwargs)
        self.bg = QtGui.QGraphicsRectItem(self)
        self.gobj = QtGui.QGraphicsSimpleTextItem(self.mobj[0].name, self.bg)        
        #self.gobj.setFont(PoolItem.font)
        if not PoolItem.fontMetrics:
            PoolItem.fontMetrics = QtGui.QFontMetrics(self.gobj.font())
        self.bg.setRect(0, 
                        0, 
                        self.gobj.boundingRect().width()
                        +PoolItem.fontMetrics.width('  '), 
                        self.gobj.boundingRect().height())
        self.bg.setPen(Qt.QColor(0,0,0,0))
        self.gobj.setPos(PoolItem.fontMetrics.width(' '), 0)

    def setDisplayProperties(self,x,y,textcolor,bgcolor):
        """Set the display properties of this item."""
        self.setGeometry(x, y,self.gobj.boundingRect().width()
                        +PoolItem.fontMetrics.width('  '), 
                        self.gobj.boundingRect().height())

        self.gobj.setPen(QtGui.QPen(QtGui.QBrush(textcolor)))
        self.gobj.setBrush(QtGui.QBrush(textcolor))
        self.bg.setBrush(QtGui.QBrush(bgcolor))
        
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable,True)
        #self.gobj.setFocus(1)
        if QtCore.QT_VERSION >= 0x040600:
            self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges,1)
    def updateSlot(self):
        self.gobj.setText(self.mobj[0].name)
        self.bg.setRect(0, 0, self.gobj.boundingRect().width()+PoolItem.fontMetrics.width('  '), self.gobj.boundingRect().height())

class ReacItem(KineticsDisplayItem):
    defaultWidth = 30
    defaultHeight = 30
    def __init__(self, *args, **kwargs):
        KineticsDisplayItem.__init__(self, *args, **kwargs)
        width = 30.0
        height = 30.0
        points = [QtCore.QPointF(ReacItem.defaultWidth/4, 0),
                  QtCore.QPointF(0,ReacItem.defaultHeight/4),
                  QtCore.QPointF(ReacItem.defaultWidth,ReacItem.defaultHeight/4),
                  QtCore.QPointF(3*ReacItem.defaultWidth/4, ReacItem.defaultHeight/2)]
        path = QtGui.QPainterPath()
        path.moveTo(points[0])
        for p in points[1:]:
            path.lineTo(p)
            path.moveTo(p)
        self.gobj = QtGui.QGraphicsPathItem(path, self)
        self.gobj.setPen(QtGui.QPen(QtCore.Qt.black, 2))
        
    def setDisplayProperties(self, x,y,textcolor,bgcolor):
        """Set the display properties of this item."""
        self.setGeometry(x,y, 
                         self.gobj.boundingRect().width(), 
                         self.gobj.boundingRect().height())
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable
                     +QtGui.QGraphicsItem.ItemIsMovable)
        if QtCore.QT_VERSION >= 0x040600:
            self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges,1)

class EnzItem(KineticsDisplayItem):
    defaultWidth = 20
    defaultHeight = 10    
    def __init__(self, *args, **kwargs):
        KineticsDisplayItem.__init__(self, *args, **kwargs)
        self.gobj = QtGui.QGraphicsEllipseItem(0, 0, 
                                            EnzItem.defaultWidth, 
                                            EnzItem.defaultHeight, self)
        
    #def setDisplayProperties(self, dinfo):
    def setDisplayProperties(self,x,y,textcolor,bgcolor):
        """Set the display properties of this item."""
        self.setGeometry(x,y, 
                         self.gobj.boundingRect().width(), 
                         self.gobj.boundingRect().height())
        self.gobj.setBrush(QtGui.QBrush(textcolor))
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable
                     +QtGui.QGraphicsItem.ItemIsMovable)
        if QtCore.QT_VERSION >= 0x040600:
            self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges,1)


class CplxItem(KineticsDisplayItem):
    defaultWidth = 10
    defaultHeight = 10    
    def __init__(self, *args, **kwargs):
        KineticsDisplayItem.__init__(self, *args, **kwargs)
        self.gobj = QtGui.QGraphicsRectItem(0, 0, CplxItem.defaultWidth, CplxItem.defaultHeight, self)

    def setDisplayProperties(self,x,y,textcolor,bgcolor):
        """Set the display properties of this item."""
        self.setGeometry(self.gobj.boundingRect().width()/2,self.gobj.boundingRect().height(), 
                         self.gobj.boundingRect().width(), 
                         self.gobj.boundingRect().height())

        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        if QtCore.QT_VERSION >= 0x040600:
            self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges,1)
        
class ComptItem(QtGui.QGraphicsRectItem):
    def __init__(self,parent,x,y,w,h,item):
        self.cmptEmitter = QtCore.QObject()
        self.mooseObj_ = item[0].parent
        self.layoutWidgetPt = parent
        QtGui.QGraphicsRectItem.__init__(self,x,y,w,h)

        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True );
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        if config.QT_MINOR_VERSION >= 6:
            self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, 1) 

    def pointerLayoutpt(self):
        return (self.layoutWidgetPt)

    def itemChange(self,change,value):
        if change == QtGui.QGraphicsItem.ItemPositionChange:
            self.cmptEmitter.emit(QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.mooseObj_)
        if change == QtGui.QGraphicsItem.ItemSelectedChange and value == True:
            self.cmptEmitter.emit(QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),self.mooseObj_)
        return QtGui.QGraphicsItem.itemChange(self,change,value)

class Widgetvisibility(Exception):pass

class  KineticsWidget(QtGui.QWidget):
    def __init__(self,size,modelPath,parent=None):
        QtGui.QWidget.__init__(self,parent)
	global itemignoreZooming
        itemignoreZooming = False
	# Get all the compartments and its members  
        cmptMol = {}
        self.setupComptObj(modelPath,cmptMol)
        #for k,v in test.items(): print k,v
        #self.itemignorestransformFlag = False
        
	#Check to see if all the cordinates are zero (which is a case for kkit8 version)
        x = []
        y = []
        allZeroCord = False
        x,y,xMin,xMax,yMin,yMax,allZeroCord = self.coordinates(x,y,cmptMol)
        
        if( allZeroCord == True):
            msgBox = QtGui.QMessageBox()
            msgBox.setText("The Layout module works for kkit version 8 or higher.")
            msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
            msgBox.exec_()
            raise Widgetvisibility()
        else:
	    #only after checking if cordiantes has values drawing starts on to qgraphicalScene
	    self.border = 10
	    hLayout = QtGui.QGridLayout(self)
	    self.setLayout(hLayout)
	    self.sceneContainer = QtGui.QGraphicsScene(self)
	    self.sceneContainer.setSceneRect(self.sceneContainer.itemsBoundingRect())
	    self.sceneContainer.setBackgroundBrush(QtGui.QColor(230,220,219,120))
            self.view = GraphicalView(self.sceneContainer,self.border,self)

            if xMax - xMin != 0:
                xratio = (size.width()-10)/(xMax-xMin)
            else:
                xratio = size.width()-10
            if yMax - yMin != 0:
                yratio = (size.height()-10)/(yMax-yMin)
            else:
                yratio = size.height()-10
                
            fnt = QtGui.QFont('Helvetica',8)

            self.cplx_width = 8

            #Compartment info goes here
            self.qGraCompt = {}
            #Map from mooseId to Graphicsobject
            self.mooseId_GObj = {}

            #pickled the color map here and loading the file
            pkl_file = open(os.path.join(config.settings[config.KEY_COLORMAP_DIR], 'rainbow2.pkl'),'rb')
            self.picklecolorMap = pickle.load(pkl_file)

            #Create compartment and its members are created on to QGraphicsScene
            self.mooseObjOntoscene(cmptMol,xratio,yratio,xMin,yMin)
            
            # Connecting lines to src and des is done here
            self.srcdesConnection = {}
            self.lineItem_dict = {}
            self.object2line = defaultdict(list)
            self.setupItem(modelPath,self.srcdesConnection)
            self.drawLine_arrow()

            hLayout.addWidget(self.view)

    def updateItemTransformationMode(self):
        global itemignoreZooming
        for v in self.sceneContainer.items():
            if( not isinstance(v,ComptItem)):
                if ( isinstance(v, PoolItem) or isinstance(v, ReacItem) or isinstance(v, EnzItem) or isinstance(v, CplxItem) ):
                    v.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations,itemignoreZooming)

    def GrVfitinView(self):
        pass
        #self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
        
    def GrViewresize(self,event):
        #when Gui resize and event is sent which inturn call resizeEvent of qgraphicsview
        self.view.resizeEvent1(event)

    def setupComptObj(self,filePath,mobject):
        ''' Compartment and its members are populated from moose '''
        cPath = filePath+'/##[TYPE=MeshEntry]'
        for meshEnt in wildcardFind(cPath):
            molrecList = []
            for reitem in Neutral(meshEnt).getNeighbors('remeshReacs'):
                molrecList.append(reitem)
            for mitem in Neutral(meshEnt).getNeighbors('remesh'):
                if ( (mitem[0].class_ != 'GslIntegrator')):
                        molrecList.append(mitem)
            mobject[meshEnt] = molrecList
    
    def coordinates(self,x,y,cmptMol):
        ''' Coordinates of each moose object retirved but cplx its enz parent is taken '''
        xMin = 0.0
        xMax = 1.0
        yMin = 0.0
        yMax = 1.0
        allzeroCord = False
        for mreObjitems in cmptMol.itervalues():
            for mre in mreObjitems:
                if ((mre[0].parent).class_ == 'ZombieEnz'):
                    mreObjinfo = (mre[0].parent).path+'/info'
                else:
                    mreObjinfo = mre.path+'/info'
                xx = float(element(mreObjinfo).getField('x'))
                yy = float(element(mreObjinfo).getField('y'))
                x.append(xx)
                y.append(yy)
        xMin= min(x)
        xMax = max(x)
        yMin = min(y)
        yMax = max(y)
        if ( len(list(self.nonzero(x))) == 0 and len(list(self.nonzero(y))) == 0  ):
            allzeroCord = True
        return(x,y,xMin,xMax,yMin,yMax,allzeroCord)    

    def nonzero(self,seq):
      return (item for item in seq if item!=0)

    def mooseObjOntoscene(self,cmptMol,xratio,yratio,xMin,yMin):
        ''' All the moose Object are put to scene '''
        for cmpt in sorted(cmptMol.iterkeys()):
            self.createCompt(cmpt)
            comptRef = self.qGraCompt[cmpt]
            mreObj = cmptMol[cmpt]
            for mre in mreObj:
                if len(mre) == 0:
                    print cmpt, " has no members in it"
                    continue
                xpos,ypos = self.positioninfo(mre,xratio,yratio,xMin,yMin)
                textcolor,bgcolor = self.colorCheck(mre,self.picklecolorMap)
                if mre.class_ == 'ZombieReac':
                    mobjItem = ReacItem(mre,comptRef)
                    mobjItem.setDisplayProperties(xpos,ypos,textcolor,bgcolor)
                elif mre.class_ =='ZombieEnz' or mre.class_ == 'ZombieMMenz':
                    mobjItem = EnzItem(mre,comptRef)
                    mobjItem.setDisplayProperties(xpos,ypos,textcolor,bgcolor)
                elif mre.class_ == 'ZombiePool' or mre.class_ == 'ZombieFuncPool' or mre.class_ == 'ZombieBufPool':
                    fnt = QtGui.QFont('Helvetica',8)
                    if mre[0].parent.class_ != 'ZombieEnz':
                        mobjItem = PoolItem(mre,comptRef)
                        #mobjItem.setfont(fnt)
                        mobjItem.setDisplayProperties(xpos,ypos,textcolor,bgcolor)
                        
                    else:
                        #cplx has made to sit under enz, for which xpos added with width/2
                        #oct4 Here I am not adding enzyme as parent for cplx
                        xpos = xpos+(self.cplx_width/2)
                        ypos = ypos
                        mobjItem = CplxItem(mre,self.mooseId_GObj[element(mre[0]).parent.getId()])
                        mobjItem.setDisplayProperties(xpos,ypos,textcolor,bgcolor)
                mobjItem.connect(mobjItem,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.positionChange)
                mobjItem.connect(mobjItem,QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),self.emitItemtoEditor)
                self.mooseId_GObj[element(mre).getId()] = mobjItem
        ''' compartment is rectangle is set '''
        for k, v in self.qGraCompt.items():
            rectcompt = v.childrenBoundingRect()
            self.comptborder = 1
            v.setRect(rectcompt.x()-10,rectcompt.y()-10,(rectcompt.width()+20),(rectcompt.height()+20))
            pen = QtGui.QPen(Qt.QColor(66,66,66,100), 10, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin)
            pen.setCosmetic(True)
            v.setPen(pen)
            v.cmptEmitter.connect(v.cmptEmitter,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.positionChange)
            v.cmptEmitter.connect(v.cmptEmitter,QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),self.emitItemtoEditor)
    
    def createCompt(self,key):
        self.new_Compt = ComptItem(self,0,0,0,0,key)
        self.qGraCompt[key] = self.new_Compt
        self.new_Compt.setRect(10,10,10,10)
        self.sceneContainer.addItem(self.new_Compt)

    def positioninfo(self,mre,xratio,yratio,xMin,yMin):
        if ((mre[0].parent).class_ == 'ZombieEnz'):
            iteminfo = (mre[0].parent).path+'/info'
        else:
            iteminfo = mre.path+'/info'

        x =  float(element(iteminfo).getField('x'))
        y = float(element(iteminfo).getField('y'))
        xpos = (x-xMin)*xratio
        ypos = -(y-yMin)*yratio
        return(xpos,ypos)

    def colorCheck(self,mre,picklecolorMap):
        if ( (mre[0]).class_ == 'ZombieEnz' or (mre[0]).class_ == 'ZombieMMenz' ):
            iteminfo = (mre[0].parent).path+'/info'
            textcolor = Annotator(iteminfo).getField('color')
            bgcolor = Annotator(iteminfo).getField('textColor')
        else:
            iteminfo = mre.path+'/info'
            textcolor = Annotator(iteminfo).getField('textColor')
            bgcolor = Annotator(iteminfo).getField('color')

        if(textcolor == ''): textcolor = 'green'
        if(bgcolor == ''): bgcolor = 'blue'
        if(textcolor == bgcolor): textcolor = self.randomColor()
        if (not isinstance(textcolor,(list,tuple))):
            if textcolor.isdigit():
                tc = int(textcolor)
                tc = (tc*2)
                textcolor = picklecolorMap[tc]
                textColor = QtGui.QColor(textcolor[0],textcolor[1],textcolor[2])
            else:
                textColor = QtGui.QColor(textcolor)
        else:
            textColor = QtGui.QColor(textcolor)
            
        if ((not isinstance(bgcolor,(list,tuple)))):
            if bgcolor.isdigit():
                tc = int(bgcolor)
                tc = (tc * 2 )
                bgcolor = picklecolorMap[tc]
                bgColor = QtGui.QColor(bgcolor[0],bgcolor[1],bgcolor[2])
            else: 
                bgColor = QtGui.QColor(bgcolor)
        else:
            bgColor = QtGui.QColor(bgcolor)
        return(textColor,bgColor)
   
    def randomColor(self):
        red = int(random.uniform(0, 255))
        green = int(random.uniform(0, 255))
        blue = int(random.uniform(0, 255))
        return (red,green,blue)
    
    def positionChange(self,mooseObject):
        #If the item position changes, the corresponding arrow's are claculated
       if ( mooseObject.class_ == 'ZombiePool' or mooseObject.class_ == 'ZombieFuncPool' or mooseObject.class_ == 'ZombieSumFunc' or mooseObject.class_ == 'ZombieBufPool'):
            pool = self.mooseId_GObj[mooseObject.getId()]
            self.updatearrow(pool)
            for k, v in self.qGraCompt.items():
                rectcompt = v.childrenBoundingRect()
                v.setRect(rectcompt.x()-10,rectcompt.y()-10,(rectcompt.width()+20),(rectcompt.height()+20))
       else:
            if((mooseObject.class_ == 'ZombieMMenz') or (mooseObject.class_ == 'ZombieEnz') or (mooseObject.class_ == 'ZombieReac') ):
                refenz = self.mooseId_GObj[mooseObject.getId()]
                self.updatearrow(refenz)
                for k, v in self.qGraCompt.items():
                    rectcompt = v.childrenBoundingRect()
                    v.setRect(rectcompt.x()-10,rectcompt.y()-10,(rectcompt.width()+20),(rectcompt.height()+20))
    
            else:
                if mooseObject.class_ == "CubeMesh":
                    for k, v in self.qGraCompt.items():
                        mesh = mooseObject.path+'/mesh[0]'
                        if k.path == mesh:
                            for rectChilditem in v.childItems():
                                self.updatearrow(rectChilditem)

    def emitItemtoEditor(self,mooseObject):
        self.emit(QtCore.SIGNAL("itemDoubleClicked(PyQt_PyObject)"),mooseObject)

    def setupItem(self,modlePath,cntDict):
        ''' Reaction's and enzyme's substrate and product and sumtotal is collected '''
        zombieType = ['ZombieReac','ZombieEnz','ZombieMMenz','ZombieSumFunc']
        for zombieObj in zombieType:
            path = modlePath+'/##[TYPE='+zombieObj+']'
            if zombieObj != 'ZombieSumFunc':
                for items in wildcardFind(path):
                    sublist = []
                    prdlist = []
                    for sub in items[0].getNeighbors('sub'): 
                        sublist.append((sub,'s'))
                    for prd in items[0].getNeighbors('prd'):
                        prdlist.append((prd,'p'))
                    if (zombieObj == 'ZombieEnz') :
                        for enzpar in items[0].getNeighbors('toEnz'):
                            sublist.append((enzpar,'t'))
                        for cplx in items[0].getNeighbors('cplxDest'):
                            prdlist.append((cplx,'cplx'))
                    if (zombieObj == 'ZombieMMenz'):
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
                    for zfun in items[0].getNeighbors('output'): funplist.append(zfun)
                    for i in funplist: nfunplist.append(element(i).getId())
                    nfunplist = list(set(nfunplist))
                    if(len(nfunplist) > 1): print "SumFunPool has multiple Funpool"
                    else:
                        for el in funplist:
                            if(element(el).getId() == nfunplist[0]):
                                cntDict[element(el)] = inputlist
                                break
    def drawLine_arrow(self):
        global itemignoreZooming
        for inn,out in self.srcdesConnection.items():
            if isinstance(out,tuple):
                if len(out[0])== 0:
                    print "Reaction or Enzyme doesn't input mssg"
                else:
                    for items in (items for items in out[0] ):
                        src = self.mooseId_GObj[inn]
                        des = self.mooseId_GObj[element(items[0]).getId()]
                        self.lineCord(src,des,items[1])
                if len(out[1]) == 0:
                    print "Reaction or Enzyme doesn't output mssg"
                else:
                    for items in (items for items in out[1] ):
                        src = self.mooseId_GObj[inn]
                        des = self.mooseId_GObj[element(items[0]).getId()]
                        self.lineCord(src,des,items[1])
            elif isinstance(out,list):
                if len(out) == 0:
                    print "Func pool doesn't have sumtotal"
                else:
                    for items in (items for items in out ):
                        src = self.mooseId_GObj[element(inn).getId()]
                        des = self.mooseId_GObj[element(items[0]).getId()]
                        self.lineCord(src,des,items[1])
    
    def lineCord(self,src,des,endtype):
        global itemignoreZooming
        source = element(next((k for k,v in self.mooseId_GObj.items() if v == src), None))
        desc = element(next((k for k,v in self.mooseId_GObj.items() if v == des), None))
        line = 0
        if (src == "") and (des == ""):
            print "Source or destination is missing or incorrect"
            return 
        srcdes_list = [src,des,endtype]        
        arrow = self.calcArrow(src,des,endtype)
        for l,v in self.object2line[src]:
                if v == des:
                    l.setPolygon(arrow)
                    return
        '''
        if itemignoreZooming:
            for l,v in self.object2line[src]:
                if v == des:
                    l.setPolygon(arrow)
                    return
        else:
            for l,v in self.object2line[src]:
                if v == des:
                    l.setPolygon(arrow)
                    return
        '''    
        qgLineitem = self.sceneContainer.addPolygon(arrow)        
        pen = QtGui.QPen(QtCore.Qt.green, 1, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin)
        pen.setCosmetic(True)
        # Green is default color moose.ReacBase and derivatives - already set above
        if  isinstance(source, moose.EnzBase):
            if ( (endtype == 's') or (endtype == 'p')):
                pen.setColor(QtCore.Qt.red)
            elif(endtype != 'cplx'):
                p = element(next((k for k,v in self.mooseId_GObj.items() if v == src), None)) 
                parentinfo = p.path+'/info'
                textColor = Annotator(parentinfo).getField('textColor')
                if(isinstance(textColor,(list,tuple))):
                    r,g,b = textColor[0],textColor[1],textColor[2]
                    color = QtGui.QColor(r,g,b)
                elif ((not isinstance(textColor,(list,tuple)))):
                    if textColor.isdigit():
                        tc = int(textColor)
                        tc = (tc * 2 )
                        r,g,b = self.picklecolorMap[tc]
                        color = QtGui.QColor(r,g,b)
                    else: 
                        color = QtGui.QColor(200,200,200)
                pen.setColor(color)
        elif isinstance(source, moose.PoolBase):
            pen.setColor(QtCore.Qt.blue)
        self.lineItem_dict[qgLineitem] = srcdes_list
        self.object2line[ src ].append( ( qgLineitem, des) )
        self.object2line[ des ].append( ( qgLineitem, src ) )
        qgLineitem.setPen(pen)
        
    def calcArrow(self,src,des,endtype):
        global itemignoreZooming
        ''' if PoolItem then boundingrect should be background rather than graphicsobject '''
        srcobj = src.gobj
        desobj = des.gobj
        if isinstance(src,PoolItem):
            srcobj = src.bg
        if isinstance(des,PoolItem):
            desobj = des.bg
        
        if itemignoreZooming:
            srcRect = self.recalcSceneBoundingRect(srcobj)
            desRect = self.recalcSceneBoundingRect(desobj)
        else:
            srcRect = srcobj.sceneBoundingRect()
            desRect = desobj.sceneBoundingRect()

        arrow = QtGui.QPolygonF()
        if srcRect.intersects(desRect):                
            # This is created for getting a emptyline for reference b'cos 
            # lineCord function add qgraphicsline to screen and also add's a ref for src and des
            arrow.append(QtCore.QPointF(0,0))
            arrow.append(QtCore.QPointF(0,0))
            return arrow
        tmpLine = QtCore.QLineF(srcRect.center().x(),
                                    srcRect.center().y(),
                                    desRect.center().x(),
                                    desRect.center().y())
        srcIntersects, lineSrcPoint = self.calcLineRectIntersection(srcRect, tmpLine)
        destIntersects, lineDestPoint = self.calcLineRectIntersection(desRect, tmpLine)
        if not srcIntersects:
            print 'Source does not intersect line. Arrow points:', lineSrcPoint, src.mobj[0].name, src.mobj[0].class_
        if not destIntersects:
            print 'Dest does not intersect line. Arrow points:', lineDestPoint,  des.mobj[0].name, des.mobj[0].class_
        # src and des are connected with line co-ordinates
        # Arrow head is drawned if the distance between src and des line is >8 just for clean appeareance
        if (abs(lineSrcPoint.x()-lineDestPoint.x()) > 8 or abs(lineSrcPoint.y()-lineDestPoint.y())>8):
            srcAngle = tmpLine.angle()
            if endtype == 'p':
                #Arrow head for Destination is calculated
                arrow.append(lineSrcPoint)
                arrow.append(lineDestPoint)
                degree = -60
                srcXArr1,srcYArr1= self.arrowHead(srcAngle,degree,lineDestPoint)
	        arrow.append(QtCore.QPointF(srcXArr1,srcYArr1))
                arrow.append(QtCore.QPointF(lineDestPoint.x(),lineDestPoint.y()))
                
		degree = -120
                srcXArr2,srcYArr2 = self.arrowHead(srcAngle,degree,lineDestPoint)
                arrow.append(QtCore.QPointF(srcXArr2,srcYArr2))                    
                arrow.append(QtCore.QPointF(lineDestPoint.x(),lineDestPoint.y()))
 
            elif endtype == 'st':
                #Arrow head for Source is calculated
                arrow.append(lineDestPoint)
                arrow.append(lineSrcPoint)
                degree = 60
                srcXArr2,srcYArr2 = self.arrowHead(srcAngle,degree,lineSrcPoint)
                arrow.append(QtCore.QPointF(srcXArr2,srcYArr2))                    
                arrow.append(QtCore.QPointF(lineSrcPoint.x(),lineSrcPoint.y()))

                degree = 120
                srcXArr1,srcYArr1= self.arrowHead(srcAngle,degree,lineSrcPoint)
		arrow.append(QtCore.QPointF(srcXArr1,srcYArr1))
                arrow.append(QtCore.QPointF(lineSrcPoint.x(),lineSrcPoint.y()))

            else:
                arrow.append(lineSrcPoint)
                arrow.append(lineDestPoint)
        return arrow
    
    def recalcSceneBoundingRect(self, rect):
        vp_trans = self.view.viewportTransform()
        trans = rect.deviceTransform(vp_trans)
        bbox = rect.boundingRect()
        sx = trans.mapRect(bbox);
        mappedRect = self.view.mapToScene(sx.toRect()).boundingRect()
        return mappedRect

    def calcLineRectIntersection(self, rect, centerLine):
        '''      checking which side of rectangle intersect with other '''
        """Here the 1. a. intersect point between center and 4 sides of src and 
        
                    b. intersect point between center and 4 sides of
                    des and to draw a line connecting for src & des
        
                    2. angle for src for the arrow head calculation is returned"""
        x = rect.x()
        y = rect.y()
        w = rect.width()
        h = rect.height()
        borders = [(x,y,x+w,y),
                   (x+w,y,x+w,y+h),
                   (x+w,y+h,x,y+h),
                   (x,y+h,x,y)]
        intersectionPoint = QtCore.QPointF()
        intersects = False
        for lineEnds in borders:
            line = QtCore.QLineF(*lineEnds)
            intersectType = centerLine.intersect(line, intersectionPoint)
            if intersectType == centerLine.BoundedIntersection:
                intersects = True
                break
        return (intersects, intersectionPoint)

    def arrowHead(self,srcAngle,degree,lineSpoint):
        '''  arrow head is calculated '''
        r = 8
        delta = math.radians(srcAngle) + math.radians(degree)
        width = math.sin(delta)*r
        height = math.cos(delta)*r
        srcXArr = lineSpoint.x() + width
        srcYArr = lineSpoint.y() + height
        return srcXArr,srcYArr
   
    def updateItemSlot(self, mooseObject):
        #In this case if the name is updated from the keyboard both in mooseobj and gui gets updation
        changedItem = ''
        for item in self.sceneContainer.items():
            if isinstance(item,PoolItem):
                if mooseObject.getId() == element(item.mobj).getId():
                    item.updateSlot()
                    #once the text is edited in editor, laydisplay gets updated in turn resize the length, positionChanged signal shd be emitted
                    self.positionChange(mooseObject)
 
    def updatearrow(self,qGTextitem):
        #if there is no arrow to update then return
        if qGTextitem not in self.object2line:
            return
        listItem = self.object2line[qGTextitem]
        for ql, va in listItem:
            srcdes = self.lineItem_dict[ql]
            if(isinstance(srcdes[0],ReacItem) or isinstance(srcdes[0],EnzItem) ):
                pItem = (next((k for k,v in self.mooseId_GObj.items() if v == srcdes[0]), None))
                mooseObj = (next((k for k,v in self.mooseId_GObj.items() if v == srcdes[1]), None))
                for l1 in self.srcdesConnection[pItem]:
                    for k in l1:
                        if ((k[0]) == mooseObj):   
                            endtype = k[1]
                        else:
                            if ( isinstance(qGTextitem,ReacItem) or isinstance(qGTextitem,EnzItem) ):
                                gItem = self.mooseId_GObj[k[0]]
                                self.updatearrow(gItem)
               
            elif(isinstance(srcdes[1],ReacItem) or isinstance(srcdes[1],EnzItem) ):
                pItem = (next((k for k,v in self.mooseId_GObj.items() if v == srcdes[1]), None))
                mooseObject = (next((k for k,v in self.mooseId_GObj.items() if v == srcdes[0]), None))
                for l1 in self.srcdesConnection[pItem]:
                    for k in l1:
                        if (k[0]) == mooseObj:
                            endtype = k[1]
            else:
                pItem  =  (next((k for k,v in self.mooseId_GObj.items() if v == srcdes[0]), None))
                pItem1 =  (next((k for k,v in self.mooseId_GObj.items() if v == srcdes[1]), None))
                if(pItem.class_ == 'ZombieFuncPool' or pItem1.class_ == 'ZombieFuncPool'):
                    endtype = 'st'
            arrow = self.calcArrow(srcdes[0],srcdes[1],endtype)
            ql.setPolygon(arrow)

    def keyPressEvent(self,event):
        key = event.key()
        global itemignoreZooming
        if key == QtCore.Qt.Key_A:
            #global itemignoreZooming

            if itemignoreZooming:
                itemignoreZooming = False
                self.updateItemTransformationMode()
                self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
                self.drawLine_arrow()
            else:
                self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)

        elif (key == 46):
            self.view
            #print '1',self.view.mapToScene(self.sceneContainer.sceneRect())
            self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.KeepAspectRatioByExpanding)
            
            
            if not itemignoreZooming:
                #self.view.scale(1.1,1.1)
                self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
                self.drawLine_arrow()
            else:
            
                itemignoreZooming = True
                self.updateItemTransformationMode()
                self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
                #self.view.scale(1.1,1.1)
                self.drawLine_arrow()
                
            
        elif (key == 44):
            if not itemignoreZooming:
                #print "1",itemignoreZooming
                self.view.scale(1/1.1,1/1.1)
                self.drawLine_arrow()
            else:
            
                itemignoreZooming = False
                self.updateItemTransformationMode()
                self.view.scale(1/1.1,1/1.1)
                self.drawLine_arrow()

        elif (key == 62):
            
            if not itemignoreZooming:
                itemignoreZooming = True
                self.updateItemTransformationMode()
                self.view.scale(1.1,1.1)
                self.drawLine_arrow()
            else:
                self.view.scale(1.1,1.1)
                self.drawLine_arrow()
        elif(key == 60):
            if not itemignoreZooming:
                itemignoreZooming = True
                self.updateItemTransformationMode()
                self.view.scale(1/1.1,1/1.1)
                self.drawLine_arrow()
            else:
                self.view.scale(1/1.1,1/1.1)
                self.drawLine_arrow()
    '''        
    def scaleItemTransformationMode(self,key,zoomInOut):
        
        global itemignoreZooming
        if key == 44 or key == 46:
            if not itemignoreZooming:
                if key == 44:
                    self.view.scale(1.1,1.1)
                else:
                    self.view.scale(1/1.1,1/1.1)
                self.drawLine_arrow()
            else:
                itemignoreZooming = False
                self.updateItemTransformationMode()
                if key == 44:
                    self.view.scale(1.1,1.1)
                else:
                    self.view.scale(1/1.1,1/1.1)

                self.drawLine_arrow()
    '''

    
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    size = QtCore.QSize(1024 ,768)
    modelPath = '77'
    modelPath = 'enz_classical_explicit'
    #modelPath = 'acc61'
    #modelPath = 'reaction4'
    #modelPath = 're'
    modelPath = 'Kholodenko'
    #modelPath = 'EGFR_MAPK_58'

    itemignoreZooming = False
    try:
        filepath = '../Demos/Genesis_files/'+modelPath+'.g'
        #filepath = '/home/harsha/genesis_files/'+modelPath+'.g'
        f = open(filepath, "r")
        loadModel(filepath,'/'+modelPath)
        dt = KineticsWidget(size,'/'+modelPath)
        dt.show()
  
    except  IOError, what:
      (errno, strerror) = what
      print "Error number",errno,"(%s)" %strerror
      sys.exit(0)
    sys.exit(app.exec_())
