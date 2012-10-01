import sys
import  os
from PyQt4 import QtGui,QtCore,Qt
import pickle
import random
import config
import re
import math
sys.path.append('../python')

from moose import *

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
        print "zoom in qgraphicview resizeevent1"
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
                if( (isinstance(sceneitems, PoolItem)) or (isinstance(sceneitems, CplxItem)) or (isinstance(sceneitems, ReacEnzItem)) ):
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
                if(isinstance(items,PoolItem) or isinstance(items, CplxItem) or isinstance(items, ReacEnzItem)):
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
        vTransform = self.viewportTransform()
        if( self.rubberbandWidth > 0  and self.rubberbandHeight >0):
            self.rubberbandlist = self.sceneContainerPt.items(self.startScenepos.x(),self.startScenepos.y(),self.rubberbandWidth,self.rubberbandHeight, Qt.Qt.IntersectsItemShape)
            for unselectitem in self.rubberbandlist:
                if unselectitem.isSelected() == True:
                    unselectitem.setSelected(0)
            for items in (qgraphicsitem for qgraphicsitem in self.rubberbandlist if isinstance(qgraphicsitem,PoolItem) or isinstance(qgraphicsitem,CplxItem) or isinstance(qgraphicsitem, ReacEnzItem)):
                self.fitInView(self.startScenepos.x(),self.startScenepos.y(),self.rubberbandWidth,self.rubberbandHeight,Qt.Qt.IgnoreAspectRatio)
                if((self.matrix().m11()>=1.0)and(self.matrix().m22() >=1.0)):
                    for item in ( Txtitem for Txtitem in self.sceneContainerPt.items() if isinstance(Txtitem,PoolItem) ):
                        item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, False)
        else:
            self.rubberbandlist = self.sceneContainerPt.items(self.endScenepos.x(),self.endScenepos.y(),abs(self.rubberbandWidth),abs(self.rubberbandHeight), Qt.Qt.IntersectsItemShape)
            for unselectitem in self.rubberbandlist:
                if unselectitem.isSelected() == True:
                    unselectitem.setSelected(0)
            for items in (qgraphicsitem for qgraphicsitem in self.rubberbandlist if isinstance(qgraphicsitem,PoolItem) or isinstance(qgraphicsitem,CplxItem) or isinstance(qgraphicsitem, ReacEnzItem)):
                self.fitInView(self.endScenepos.x(),self.endScenepos.y(),abs(self.rubberbandWidth),abs(self.rubberbandHeight),Qt.Qt.IgnoreAspectRatio)
                if((self.matrix().m11()>=1.0)and(self.matrix().m22() >=1.0)):
                    for item in ( Txtitem for Txtitem in self.sceneContainerPt.items() if isinstance (Txtitem, PoolItem)):
                        item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, False)
        self.rubberBandactive = False



class CplxItem(QtGui.QGraphicsRectItem):
    def __init__(self,parent,x,y,w,h,item):
        self.cplxEmitter = QtCore.QObject()
        self.mooseObj_ = item
        self.layoutWidgetPt = parent

        QtGui.QGraphicsRectItem.__init__(self,x,y,w,h,parent)

        #self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)

        #~ self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, 1) 
        if config.QT_MINOR_VERSION >= 6:
            self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, 1) 

    def pointerLayoutpt(self):
        return (self.layoutWidgetPt)
    
    def mouseDoubleClickEvent(self, event):
        self.cplxEmitter.emit(QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),element(self.mooseObj_))
    
    def itemChange(self,change,value):
        if change == QtGui.QGraphicsItem.ItemPositionChange:
            self.cplxEmitter.emit(QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),element(self.mooseObj_))
        if change == QtGui.QGraphicsItem.ItemSelectedChange and value == True:
            self.cplxEmitter.emit(QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),element(self.mooseObj_))
        return QtGui.QGraphicsItem.itemChange(self,change,value)

class PoolItem(QtGui.QGraphicsSimpleTextItem):
    def __init__(self,parent,item,xpos,ypos,pickle):
        self.mooseObj_ = item[0]
        iteminfo = item.path+'/info'
        textcolor = Annotator(iteminfo).getField('textColor')
        bgcolor = Annotator(iteminfo).getField('color')
        self.picklecolorMap = pickle
        self.parent = parent
        self.poolEmitter = QtCore.QObject()
        if isinstance(parent,KineticsWidget):
            QtGui.QGraphicsSimpleTextItem.__init__(self,self.mooseObj_.name)
            self.layoutWidgetpt = parent
        elif isinstance(parent,ComptItem):
            self.layoutWidgetpt = parent.pointerLayoutpt()
            QtGui.QGraphicsSimpleTextItem.__init__(self, self.mooseObj_.name, self.parent)

        textcolor,bgcolor = self.layoutWidgetpt.colorCheck(textcolor,bgcolor,self.picklecolorMap)
        if(isinstance(textcolor,(list,tuple))):
            r,g,b = textcolor[0],textcolor[1],textcolor[2]
            self.textColor = QtGui.QColor(r,g,b)
        else:
            self.textColor = QtGui.QColor(textcolor)

        if(isinstance(bgcolor,(list,tuple))):
            r,g,b = bgcolor[0],bgcolor[1],bgcolor[2]
            self.bgColor = QtGui.QColor(r,g,b)
        else:
            self.bgColor = QtGui.QColor(bgcolor)
        self.setPen(QtGui.QPen(QtGui.QBrush(Qt.Qt.black)))
        self.setBrush(QtGui.QBrush(self.textColor))
        self.setPos(xpos,ypos)
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        
        if config.QT_MINOR_VERSION >= 6:
            self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, 1) 

    def paint(self, painter, option, widget):
        painter.setPen(Qt.QColor(0,0,0,0))
        painter.setBrush(QtGui.QBrush(self.bgColor))
        painter.drawRect(self.boundingRect().x(),self.boundingRect().y(),self.boundingRect().width(),self.boundingRect().height())
        QtGui.QGraphicsSimpleTextItem.paint(self, painter, option, widget)

    def mouseDoubleClickEvent(self, event):
        self.poolEmitter.emit(QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),element(self.mooseObj_))
    
    def updateSlot(self):
        self.setText(self.mooseObj_.name)
    
    def itemChange(self,change,value):
        if change == QtGui.QGraphicsItem.ItemPositionChange:
            self.poolEmitter.emit(QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.mooseObj_)
        if change == QtGui.QGraphicsItem.ItemSelectedChange and value == True:
            self.poolEmitter.emit(QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),self.mooseObj_)
        return QtGui.QGraphicsItem.itemChange(self,change,value)

class ReacEnzItem(QtGui.QGraphicsEllipseItem):
    #ositionChange = QtCore.pyqtSignal(QtGui.QGraphicsItem)
    #selectedChange = QtCore.pyqtSignal(QtGui.QGraphicsItem)
    def __init__(self,x,y,w,h,parent,mooseObj):
        self.reacenzEmitter = QtCore.QObject()
        self.mooseObj_ = mooseObj[0]
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.layoutWidgetPt = parent
        if isinstance(parent,KineticsWidget):
            QtGui.QGraphicsEllipseItem.__init__(self,x,y,w,h)
        elif isinstance(parent,ComptItem):
            QtGui.QGraphicsEllipseItem.__init__(self,x,y,w,h,parent)
            self.layoutWidgetpt = parent.pointerLayoutpt()
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True );        
        self.setFlag(QtGui.QGraphicsItem.ItemSendsScenePositionChanges, True );
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
    
    def pointerLayoutpt(self):
        return (self.layoutWidgetPt)
    def mouseDoubleClickEvent(self, event):
        self.reacenzEmitter.emit(QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),element(self.mooseObj_))
    
    def itemChange(self,change,value):
        if change == QtGui.QGraphicsItem.ItemPositionChange:
            self.reacenzEmitter.emit(QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),element(self.mooseObj_))
        if change == QtGui.QGraphicsItem.ItemSelectedChange and value == True:
            self.reacenzEmitter.emit(QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),element(self.mooseObj_))
        return QtGui.QGraphicsItem.itemChange(self,change,value)
    
    def paint(self, painter, option, widget):
        if(self.mooseObj_.class_ == 'ZombieReac'):
            x = self.x
            y = self.y
            width = self.sceneBoundingRect().width()
            height = self.sceneBoundingRect().height()
            painter.setPen(QtGui.QPen(QtGui.QBrush(Qt.Qt.black),1))
            arrow = QtGui.QPolygonF([QtCore.QPointF(x,y+height/2),QtCore.QPointF(x+width,y+height/2)])
            painter.drawPolygon(arrow)
            arrow = QtGui.QPolygonF([QtCore.QPointF(x,y+height/2),QtCore.QPointF(x+(math.sin(90)*8),y+height/2+math.cos(90)*8)])
            painter.drawPolygon(arrow)
            arrow = QtGui.QPolygonF([QtCore.QPointF(x+width,y+height/2),QtCore.QPointF(x+width+(math.sin(225)*8),y+height/2+math.cos(225)*8)])
            painter.drawPolygon(arrow)
        else:
            QtGui.QGraphicsEllipseItem.paint(self,painter,option,widget)

class ComptItem(QtGui.QGraphicsRectItem):
    def __init__(self,parent,x,y,w,h,item):
        self.cmptEmitter = QtCore.QObject()
        self.mooseObj_ = item[0].parent
        self.layoutWidgetPt = parent
        QtGui.QGraphicsRectItem.__init__(self,x,y,w,h)
        '''
        if isinstance(parent,KineticsWidget):
            QtGui.QGraphicsRectItem.__init__(self,x,y,w,h)

        elif isinstance(parent,Item):
            QtGui.QGraphicsRectItem.__init__(self,x,y,w,h,parent)
            self.layoutWidgetpt = parent.pointerLayoutpt()
        '''
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True );
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        if config.QT_MINOR_VERSION >= 6:
            self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, 1) 

    def pointerLayoutpt(self):
        return (self.layoutWidgetPt)
    
    def mouseDoubleClickEvent(self, event):
        self.cmptEmitter.emit(QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.mooseObj_)
    
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
	
	# Get all the compartments and its members  
        cmptMol = {}
        self.setupComptObj(modelPath,cmptMol)
        #for k,v in test.items(): print k,v

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
		
            if xMax - xMin != 0:
                xratio = (size.width()-10)/(xMax-xMin)
            else:
                xratio = size.width()-10
            if yMax - yMin != 0:
                yratio = (size.height()-10)/(yMax-yMin)
            else:
                yratio = size.height()-10
                
            fnt = QtGui.QFont('Helvetica',8)

            #pickled the color map here and loading the file
	    pkl_file = open(os.path.join(config.settings[config.KEY_COLORMAP_DIR], 'rainbow2.pkl'),'rb')
            self.picklecolorMap = pickle.load(pkl_file)        

            self.qGraCompt = {}
            self.mooseId_GText = {}
            self.ellipse_width = 15
            self.ellipse_height = 15
            self.cplx_width = 8
            self.cplx_height = 8
            for cmpt in sorted(cmptMol.iterkeys()):
                self.createCompt(cmpt)
                comptRef = self.qGraCompt[cmpt]
                mreObj = cmptMol[cmpt]
                for mre in mreObj:
                    if len(mre) == 0:
                        continue
                    if ((mre[0].parent).class_ == 'ZombieEnz'):
                        mreinfo = (mre[0].parent).path+'/info'
                    else:
                        mreinfo = mre.path+'/info'
                    x =  float(element(mreinfo).getField('x'))
                    y = float(element(mreinfo).getField('y'))
                    xpos = (x-xMin)*xratio
                    ypos = -(y-yMin)*yratio
                    if mre.class_ == 'ZombieEnz' or mre.class_ == 'ZombieMMenz' or mre.class_ == 'ZombieReac':
                        if mre.class_ == 'ZombieReac':
                            reItem = ReacEnzItem(xpos,ypos,self.ellipse_width,self.ellipse_height,comptRef,mre)
                        elif mre.class_ =='ZombieEnz' or mre.class_ == 'ZombieMMenz':
                            reItem = ReacEnzItem(xpos,ypos,self.ellipse_width,self.ellipse_height,comptRef,mre)
                            mreObjinfo = (mre[0].parent).path+'/info'
                            textcolor = ''
                            bgcolor = Annotator(mreObjinfo).getField('color')
                            textcolor,bgcolor = self.colorCheck(textcolor,bgcolor,self.picklecolorMap)
                            if(isinstance(bgcolor,(list,tuple))):
                                r,g,b = bgcolor[0],bgcolor[1],bgcolor[2]
                                self.bcolor = QtGui.QColor(r,g,b)
                            else:
                                self.bcolor = QtGui.QColor(bgcolor)
                            reItem.setBrush(QtGui.QColor(self.bcolor))
                            
                        reItem.reacenzEmitter.connect(reItem.reacenzEmitter, QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.emitItemtoEditor)
                        reItem.reacenzEmitter.connect(reItem.reacenzEmitter,QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),self.emitItemtoEditor)
                        reItem.reacenzEmitter.connect(reItem.reacenzEmitter,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.positionChange)
                        self.mooseId_GText[element(mre).getId()] = reItem
                    elif mre.class_ == 'ZombiePool' or mre.class_ == 'ZombieFuncPool' or mre.class_ == 'ZombieBufPool':
                        if mre[0].parent.class_ != 'ZombieEnz':
                            pItem = PoolItem(comptRef,mre,xpos,ypos,self.picklecolorMap)
                            pItem.setFont(fnt)
                            pItem.poolEmitter.connect(pItem.poolEmitter, QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.emitItemtoEditor)
                            pItem.poolEmitter.connect(pItem.poolEmitter,QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),self.emitItemtoEditor)
                            pItem.poolEmitter.connect(pItem.poolEmitter,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.positionChange)
                        else:
                            #cplx has made to sit under enz, for which xpos added with width/2 and height is added by ellipseitem height which is 15 for now
                            xpos = xpos+(self.cplx_width/2)
                            ypos = ypos+self.ellipse_width
                            pItem = CplxItem(self.mooseId_GText[element(mre[0]).parent.getId()],xpos,ypos,self.cplx_width,self.cplx_height,mre[0])
                            textcolor = ''
                            pItem.cplxEmitter.connect(pItem.cplxEmitter,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.positionChange)
                            pItem.cplxEmitter.connect(pItem.cplxEmitter,QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.emitItemtoEditor)
                            pItem.cplxEmitter.connect(pItem.cplxEmitter,QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),self.emitItemtoEditor)
                        self.mooseId_GText[element(mre).getId()] = pItem
            for k, v in self.qGraCompt.items():
                rectcompt = v.childrenBoundingRect()
                v.setRect(rectcompt.x()-10,rectcompt.y()-10,(rectcompt.width()+20),(rectcompt.height()+20))
                v.setPen(QtGui.QPen(Qt.QColor(66,66,66,100),10,QtCore.Qt.SolidLine,QtCore.Qt.RoundCap,QtCore.Qt.RoundJoin ))
                v.cmptEmitter.connect(v.cmptEmitter,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.positionChange)
                v.cmptEmitter.connect(v.cmptEmitter,QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.emitItemtoEditor)                
                v.cmptEmitter.connect(v.cmptEmitter,QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),self.emitItemtoEditor)
            ## Connections b/w pools and reaction and enz
            self.srcdesConnection = {}
            self.lineItem_dict = {}
            self.object2line = {}
            self.setupItem(modelPath,self.srcdesConnection)
            #for p,q in self.srcdesConnection.items():    print p,q
            for inn,out in self.srcdesConnection.items():
                if isinstance(out,tuple):
                    if len(out[0])== 0:
                        print "Reaction or Enzyme doesn't input mssg"
                    else:
                        for items in (items for items in out[0] ):
                            src = ""
                            des = ""
                            src = self.mooseId_GText[inn]
                            des = self.mooseId_GText[element(items[0]).getId()]
                            self.lineCord(src,des,items[1])
                    if len(out[1]) == 0:
                        print "Reaction or Enzyme doesn't output mssg"
                    else:
                        for items in (items for items in out[1] ):
                            src = ""
                            des = ""
                            src = self.mooseId_GText[inn]
                            des = self.mooseId_GText[element(items[0]).getId()]
                            self.lineCord(src,des,items[1])
                elif isinstance(out,list):
                    if len(out) == 0:
                        print "Func pool doesn't have sumtotal"
                    else:
                         for items in (items for items in out ):
                             src = ""
                             des = ""
                             src = self.mooseId_GText[element(inn).getId()]
                             des = self.mooseId_GText[element(items[0]).getId()]
                             self.lineCord(src,des,items[1])
                    
            self.view = GraphicalView(self.sceneContainer,self.border,self)
            self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
            hLayout.addWidget(self.view)

    def GrVfitinView(self):
        self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
    
    def updateItemSlot(self, mooseObject):
        #In this case if the name is updated from the keyboard both in mooseobj and gui gets updation
        changedItem = ''
        for changedItem in (item for item in self.sceneContainer.items() if isinstance(item, PoolItem) and mooseObject.getId() == item.mooseObj_.getId()):
            break
        
        if isinstance(changedItem,PoolItem):
            changedItem.updateSlot()
            #once the text is edited in editor, width gets resized for the positionChange signal shd be emitted"
            self.positionChange(changedItem.mooseObj_)

    def updatearrow(self,qGTextitem):
        #if there is no arrow to update then return
        if qGTextitem not in self.object2line:
            return
        listItem = self.object2line[qGTextitem]
        for ql, va in listItem:
            srcdes = self.lineItem_dict[ql]
            forloop = 0
            if(isinstance(srcdes[0],ReacEnzItem)):
                pItem = (next((k for k,v in self.mooseId_GText.items() if v == srcdes[0]), None))
                mooseObj = (next((k for k,v in self.mooseId_GText.items() if v == srcdes[1]), None))
                for l1 in self.srcdesConnection[pItem]:
                    for k in l1:
                        if ((k[0]) == mooseObj):   
                            endtype = k[1]
                        else:
                            if isinstance(qGTextitem,ReacEnzItem):
                                gItem = self.mooseId_GText[k[0]]
                                self.updatearrow(gItem)
               
            elif(isinstance(srcdes[1],ReacEnzItem)):
                pItem = (next((k for k,v in self.mooseId_GText.items() if v == srcdes[1]), None))
                mooseObject = (next((k for k,v in self.mooseId_GText.items() if v == srcdes[0]), None))
                for l1 in self.srcdesConnection[pItem]:
                    for k in l1:
                        if (k[0]) == mooseObj:
                            endtype = k[1]
            else:
                pItem  =  (next((k for k,v in self.mooseId_GText.items() if v == srcdes[0]), None))
                pItem1 =  (next((k for k,v in self.mooseId_GText.items() if v == srcdes[1]), None))
                if(pItem.class_ == 'ZombieFuncPool' or pItem1.class_ == 'ZombieFuncPool'):
                    endtype = 'st'

            arrow = self.calArrow(srcdes[0],srcdes[1],endtype)
            ql.setPolygon(arrow)

    def coordinates(self,x,y,cmptMol):
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

    def calArrow(self,src,des,endtype):
        
        sX = src.sceneBoundingRect().x()
        sY = src.sceneBoundingRect().y()
        sw = src.sceneBoundingRect().right() -src.sceneBoundingRect().left()
        sh = src.sceneBoundingRect().bottom() -src.sceneBoundingRect().top()
        
        dX = des.sceneBoundingRect().x()
        dY = des.sceneBoundingRect().y()
        dw = des.sceneBoundingRect().right() -des.sceneBoundingRect().left()
        dh = des.sceneBoundingRect().bottom() -des.sceneBoundingRect().top()

        #Here there is external boundary created for each textitem 
        #1. for checking if there is overLap
        #2. The start line and arrow head ends to this outer boundary
        
        srcRect = src.sceneBoundingRect()
        desRect = des.sceneBoundingRect()
        #To see if the 2 boundary rect of text item intersects
        t = srcRect.intersects(desRect)
        arrow = QtGui.QPolygonF()
        if not t:
            centerPoint = QtCore.QLineF(src.sceneBoundingRect().center().x(),src.sceneBoundingRect().center().y(),des.sceneBoundingRect().center().x(),des.sceneBoundingRect().center().y())
            lineSrcpoint = QtCore.QPointF(0,0)
            srcAngle = self.calPoAng(sX,sY,sw,sh,centerPoint,lineSrcpoint)
            lineDespoint = QtCore.QPointF(0,0)
            self.calPoAng(dX,dY,dw,dh,centerPoint,lineDespoint)
            # src and des are connected with line co-ordinates
            #Arrow head is drawned if the distance between src and des line is >8 just for clean appeareance
            if endtype == 'p':
                if(abs(lineSrcpoint.x()-lineDespoint.x()) > 8 or abs(lineSrcpoint.y()-lineDespoint.y())>8):
                    #Arrow head for Source is calculated
                    srcAngle = self.calPoAng(dX,dY,dw,dh,centerPoint,lineDespoint)
                    degree = -60
                    srcXArr1,srcYArr1= self.arrowHead(srcAngle,degree,lineDespoint)
                    degree = -120
                    srcXArr2,srcYArr2 = self.arrowHead(srcAngle,degree,lineDespoint)
                    arrow.append(QtCore.QPointF(lineSrcpoint.x(),lineSrcpoint.y()))
                    arrow.append(QtCore.QPointF(lineDespoint.x(),lineDespoint.y()))
                    
                    arrow.append(QtCore.QPointF(srcXArr1,srcYArr1))
                    arrow.append(QtCore.QPointF(lineDespoint.x(),lineDespoint.y()))
                    arrow.append(QtCore.QPointF(srcXArr2,srcYArr2))                    
                    arrow.append(QtCore.QPointF(lineDespoint.x(),lineDespoint.y()))
            elif endtype == 'st':
                if(abs(lineSrcpoint.x()-lineDespoint.x()) > 8 or abs(lineSrcpoint.y()-lineDespoint.y())>8):
                    #Arrow head for Source is calculated
                    desAngle = self.calPoAng(sX,sY,sw,sh,centerPoint,lineSrcpoint)
                    degree = 120
                    srcXArr1,srcYArr1= self.arrowHead(srcAngle,degree,lineSrcpoint)
                    degree = 60
                    srcXArr2,srcYArr2 = self.arrowHead(srcAngle,degree,lineSrcpoint)
                    arrow.append(QtCore.QPointF(lineDespoint.x(),lineDespoint.y()))
                    arrow.append(QtCore.QPointF(lineSrcpoint.x(),lineSrcpoint.y()))
                    
                    arrow.append(QtCore.QPointF(srcXArr1,srcYArr1))
                    arrow.append(QtCore.QPointF(lineSrcpoint.x(),lineSrcpoint.y()))
                    arrow.append(QtCore.QPointF(srcXArr2,srcYArr2))                    
                    arrow.append(QtCore.QPointF(lineSrcpoint.x(),lineSrcpoint.y()))
            else:
                arrow.append(QtCore.QPointF(lineSrcpoint.x(),lineSrcpoint.y()))
                arrow.append(QtCore.QPointF(lineDespoint.x(),lineDespoint.y()))
                                    
            return (arrow)
        elif t:
            # This is created for getting a emptyline for reference b'cos 
            # lineCord function add qgraphicsline to screen and also add's a ref for src and des
            arrow.append(QtCore.QPointF(0,0))
            arrow.append(QtCore.QPointF(0,0))
            return (arrow)

    #checking which side of rectangle intersect with other
    def calPoAng(self,X,Y,w,h,centerLine,linePoint):
            #Here the 1. a. intersect point between center and 4 sides of src and 
            #            b. intersect point between center and 4 sides of des and to draw a line connecting for src & des
            #         2. angle for src for the arrow head calculation is returned
            #lItemSP = QtCore.QLineF(X-5,Y-5,X+w+5,Y-5)
            lItemSP = QtCore.QLineF(X,Y,X+w,Y)
            boundintersect= lItemSP.intersect(centerLine,linePoint)
            if (boundintersect == 1):
                return centerLine.angle()
            else:
                #lItemSP = QtCore.QLineF(X+w+5,Y-5,X+w+5,Y+h+5)
                lItemSP = QtCore.QLineF(X+w,Y,X+w,Y+h)
                boundintersect= lItemSP.intersect(centerLine,linePoint)
                if (boundintersect == 1):
                    return centerLine.angle()
                else:
                    #lItemSP = QtCore.QLineF(X+w+5,Y+h+5,X-5,Y+h+5)
                    lItemSP = QtCore.QLineF(X+w,Y+h,X,Y+h)
                    boundintersect= lItemSP.intersect(centerLine,linePoint)
                    if (boundintersect == 1):
                        return centerLine.angle()
                    else:
                        #lItemSP = QtCore.QLineF(X-5,Y+h+5,X-5,Y-5)
                        lItemSP = QtCore.QLineF(X,Y+h,X,Y)
                        boundintersect= lItemSP.intersect(centerLine,linePoint)
                        if (boundintersect == 1):
                            return centerLine.angle()
                        else:
                            linePoint = QtCore.QPointF(0,0)
                            return 0

    #arrow head is calculated
    def arrowHead(self,srcAngle,degree,lineSpoint):
        r = 8
        delta = math.radians(srcAngle) + math.radians(degree)
        width = math.sin(delta)*r
        height = math.cos(delta)*r
        srcXArr = lineSpoint.x() + width
        srcYArr = lineSpoint.y() + height
        return srcXArr,srcYArr

    def lineCord(self,src,des,endtype):
        source = element(next((k for k,v in self.mooseId_GText.items() if v == src), None))
        desc = element(next((k for k,v in self.mooseId_GText.items() if v == des), None))
        line = 0
        if( (src == "") and (des == "") ):
            print "Source or destination is missing or incorrect"
        else:
            srcdes_list= [src,des,endtype]
            arrow = self.calArrow(src,des,endtype)
            if(source.class_ == "ZombieReac"):
                qgLineitem = self.sceneContainer.addPolygon(arrow,QtGui.QPen(QtCore.Qt.green, 1, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
                line = 1
            elif( (source.class_ == "ZombieEnz") or (source.class_ == "ZombieMMenz")):
                
                if ( (endtype == 's') or (endtype == 'p')):
                    qgLineitem = self.sceneContainer.addPolygon(arrow,QtGui.QPen(QtCore.Qt.red, 1, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
                    line = 1
                elif(endtype != 'cplx'):
                    p = element(next((k for k,v in self.mooseId_GText.items() if v == src), None)) 
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
                    qgLineitem = self.sceneContainer.addPolygon(arrow,QtGui.QPen(color ,1, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
                    line = 1
                elif(endtype == 'cplx'):
                    pass
            elif( (source.class_ == "ZombiePool") or (source.class_ == "ZombieFuncPool") or (source.class_ == "ZombieBuffPool")):
                qgLineitem = self.sceneContainer.addPolygon(arrow,QtGui.QPen(QtCore.Qt.blue, 1, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
                line =1
            if line == 1:            
                self.lineItem_dict[qgLineitem] = srcdes_list
                if src in self.object2line:
                    self.object2line[ src ].append( ( qgLineitem, des) )
                else:
                    self.object2line[ src ] = []
                    self.object2line[ src ].append( ( qgLineitem, des) )
                if des in self.object2line:
                    self.object2line[ des ].append( ( qgLineitem, src ) )
                else:
                    self.object2line[ des ] = []
                    self.object2line[ des ].append( ( qgLineitem, src) )

    def setupItem(self,modlePath,cntDict):
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

    def GrViewresize(self,event):
        self.view.resizeEvent1(event)    
    
    
    def keyPressEvent(self,event):
        key = event.key()
        if key == QtCore.Qt.Key_A:
            #pass
            self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
        elif (key == 46 or key == 62):
            self.view.scale(1.1,1.1)
        elif (key == 44 or key == 60):
            self.view.scale(1/1.1,1/1.1)

    def positionChange(self,mooseObject):
        #If the item position changes, the corresponding arrow's are claculated
       if ( mooseObject.class_ == 'ZombiePool' or mooseObject.class_ == 'ZombieFuncPool' or mooseObject.class_ == 'ZombieSumFunc' or mooseObject.class_ == 'ZombieBufPool'):
            pool = self.mooseId_GText[mooseObject.getId()]
            self.updatearrow(pool)
            for k, v in self.qGraCompt.items():
                rectcompt = v.childrenBoundingRect()
                v.setRect(rectcompt.x()-10,rectcompt.y()-10,(rectcompt.width()+20),(rectcompt.height()+20))
       else:
            if((mooseObject.class_ == 'ZombieMMenz') or (mooseObject.class_ == 'ZombieEnz') or (mooseObject.class_ == 'ZombieReac') ):
                refenz = self.mooseId_GText[mooseObject.getId()]
                self.updatearrow(refenz)
                for k, v in self.qGraCompt.items():
                    rectcompt = v.childrenBoundingRect()
                    v.setRect(rectcompt.x()-10,rectcompt.y()-10,(rectcompt.width()+20),(rectcompt.height()+20))
    
            else:
                print "mooseObj",mooseObject.class_
                if mooseObject.class_ == "CubeMesh":
                    for k, v in self.qGraCompt.items():
                        mesh = mooseObject.path+'/mesh[0]'
                        if k.path == mesh:
                            for rectChilditem in v.childItems():
                                self.updatearrow(rectChilditem)

    def emitItemtoEditor(self,mooseObject):
        self.emit(QtCore.SIGNAL("itemDoubleClicked(PyQt_PyObject)"),mooseObject)

    def colorCheck(self,textColor,bgcolor,pklcolor):
        if(textColor == ''): textColor = 'green'
        if(bgcolor == ''): bgcolor = 'blue'
        if(textColor == bgcolor): textColor = self.randomColor()
        if (not isinstance(textColor,(list,tuple))):
            if textColor.isdigit():
                tc = int(textColor)
                tc = (tc*2)
                textColor = pklcolor[tc]
        if ((not isinstance(bgcolor,(list,tuple)))):
            if bgcolor.isdigit():
                tc = int(bgcolor)
                tc = (tc * 2 )
                bgcolor = pklcolor[tc]
        #print "here",textColor,bgcolor
        return(textColor,bgcolor)
   
    def randomColor(self):
        red = int(random.uniform(0, 255))
        green = int(random.uniform(0, 255))
        blue = int(random.uniform(0, 255))
        return (red,green,blue)            
    def createCompt(self,key):
        self.new_Compt = ComptItem(self,0,0,0,0,key)
        self.qGraCompt[key] = self.new_Compt
        self.new_Compt.setRect(10,10,10,10)
        self.sceneContainer.addItem(self.new_Compt)
    
    def setupComptObj(self,filePath,mobject):
        cPath = filePath+'/##[TYPE=MeshEntry]'
        for meshEnt in wildcardFind(cPath):
            molrecList = []
            for reitem in Neutral(meshEnt).getNeighbors('remeshReacs'):
                molrecList.append(reitem)
            for mitem in Neutral(meshEnt).getNeighbors('remesh'):
                if ( (mitem[0].class_ != 'GslIntegrator')):
                        molrecList.append(mitem)
            mobject[meshEnt] = molrecList

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    size = QtCore.QSize(1024 ,768)
    modelPath = 'Kholodenko'
    #modelPath = 'acc61'
  
    try:
        filepath = '../Demos/Genesis_files/'+modelPath+'.g'
        f = open(filepath, "r")
        loadModel(filepath,'/'+modelPath)
        dt = KineticsWidget(size,'/'+modelPath)
        dt.show()
  
    except  IOError, what:
      (errno, strerror) = what
      print "Error number",errno,"(%s)" %strerror
      sys.exit(0)
    sys.exit(app.exec_())
