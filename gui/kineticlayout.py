#This file is for loading Genesis file to mooseGui

import sys
import os
from PyQt4 import QtGui,QtCore,Qt
import pygraphviz as pgv
import pickle
import random
import config
import re
import math

from filepaths import *

from moose import *

class polygonItem(QtGui.QGraphicsPolygonItem):
    #positionChange = QtCore.pyqtSignal(QtGui.QGraphicsItem)
    #selectedChange = QtCore.pyqtSignal(QtGui.QGraphicsItem)
    def __init__(self,polygon,parent,item):
        self.polyemitter = QtCore.QObject()
        self.mooseObj_ = item
        if isinstance(parent,KineticsWidget):
            QtGui.QGraphicsPolygonItem(polygon)
        elif isinstance(parent,RectCompt):
            QtGui.QGraphicsPolygonItem(polygon,parent)

    def mouseDoubleClickEvent(self, event):
        self.polyemitter.emit(QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.mooseObj_)

class EllipseItem(QtGui.QGraphicsEllipseItem):
    positionChange = QtCore.pyqtSignal(QtGui.QGraphicsItem)
    selectedChange = QtCore.pyqtSignal(QtGui.QGraphicsItem)

    def __init__(self,x,y,w,h,parent,item):
        self.ellemitter = QtCore.QObject()
        self.mooseObj_ = item
        if isinstance(parent,KineticsWidget):
            QtGui.QGraphicsEllipseItem.__init__(self,x,y,w,h)
        elif isinstance(parent,RectCompt):
            QtGui.QGraphicsEllipseItem.__init__(self,x,y,w,h,parent)

        
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable, True );
        self.setFlag(QtGui.QGraphicsItem.ItemSendsScenePositionChanges, True );
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)

    def mouseDoubleClickEvent(self, event):
        self.ellemitter.emit(QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.mooseObj_)
    
    def itemChange(self,change,value):
        if change == QtGui.QGraphicsItem.ItemPositionChange:
            self.ellemitter.emit(QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.mooseObj_)
        return QtGui.QGraphicsItem.itemChange(self,change,value)

class RectCompt(QtGui.QGraphicsRectItem):
    def __init__(self,parent,x,y,w,h,item):
        self.Rectemitter = QtCore.QObject()
        self.mooseObj_ = item.parent
        self.layoutWidgetPt = parent
        #if isinstance(parent,kineticsWidget):
        QtGui.QGraphicsRectItem.__init__(self,x,y,w,h)
        
        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)

        #~ self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, 1) 
        if config.QT_MINOR_VERSION >= 6:
            self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, 1) 

    def pointerLayoutpt(self):
        return (self.layoutWidgetPt)
    
    def mouseDoubleClickEvent(self, event):
        self.Rectemitter.emit(QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.mooseObj_)
    
    def itemChange(self,change,value):
        if change == QtGui.QGraphicsItem.ItemPositionChange:
            self.Rectemitter.emit(QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.mooseObj_)
        return QtGui.QGraphicsItem.itemChange(self,change,value)

class Textitem(QtGui.QGraphicsTextItem):
    positionChange = QtCore.pyqtSignal(QtGui.QGraphicsItem)
    selectedChange = QtCore.pyqtSignal(QtGui.QGraphicsItem)

    def __init__(self,parent,mooseObj):
        self.mooseObj_ = mooseObj
        if isinstance(parent,KineticsWidget):
            QtGui.QGraphicsTextItem.__init__(self,mooseObj.name)
            self.layoutWidgetpt = parent
        elif isinstance(parent,RectCompt):
            self.layoutWidgetpt = parent.pointerLayoutpt()
            QtGui.QGraphicsTextItem.__init__(self,parent)

        self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
        
        if config.QT_MINOR_VERSION >= 6:
            self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, 1) 
        
    def mouseDoubleClickEvent(self, event):
        self.emit(QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.mooseObj_)
    
    def updateSlot(self):
        iteminfo = self.mooseObj_.path+'/info'
        pkl_file = open(os.path.join(PATH_KKIT_COLORMAPS,'rainbow2.pkl'),'rb')
        picklecolorMap = pickle.load(pkl_file)
        if((self.mooseObj_.class_ =='ZombieEnz') or (self.mooseObj_.class_ =='ZombieMMenz')):
            textbgcolor = 'black'
            textcolor = ''
            textcolor = Annotator(iteminfo).getField('color')
        else:
            textbgcolor = Annotator(iteminfo).getField('color')
            textcolor = Annotator(iteminfo).getField('textColor')
        
        textcolor,textbgcolor = self.layoutWidgetpt.colorCheck(self.mooseObj_,textcolor,textbgcolor,picklecolorMap)
        self.setDefaultTextColor(QtGui.QColor(textcolor))
        textbgcolor = "<html><body bgcolor='"+textbgcolor+"'>"+self.mooseObj_.name+"</body></html>"
        self.setHtml(QtCore.QString(textbgcolor))
    
    def itemChange(self,change,value):
        if change == QtGui.QGraphicsItem.ItemPositionChange:
            self.positionChange.emit(self)
        if change == QtGui.QGraphicsItem.ItemSelectedChange and value == True:
           self.selectedChange.emit(self)
        return QtGui.QGraphicsItem.itemChange(self,change,value)

class GraphicalView(QtGui.QGraphicsView):
    def __init__(self,parent,border):
        QtGui.QGraphicsView.__init__(self,parent)
        #print '$$',parent,parent.sceneRect()
        self.setScene(parent)
        self.sceneContainerPt = parent
        self.setDragMode(QtGui.QGraphicsView.RubberBandDrag)
        self.itemSelected = False
        self.customrubberBand=0
        self.rubberbandWidth = 0
        self.rubberbandHeight = 0
        self.moved = False
        self.showpopupmenu = False
        self.border = 10
    ''' Need to check resize evenet v/s fitinview '''
    '''
    def resizeEvent(self, event):
        """ zoom when resize! """
        self.fitInView(self.sceneContainerPt.sceneRect(), Qt.Qt.IgnoreAspectRatio)
        QtGui.QGraphicsView.resizeEvent(self, event)
   ''' 
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
            elif( sceneitems != None):
                if ( (isinstance(sceneitems, Textitem)) or (isinstance(sceneitems, EllipseItem))):
                    QtGui.QGraphicsView.mousePressEvent(self, event)
                    self.itemSelected = True

                elif(isinstance(sceneitems, RectCompt)):
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
                if((isinstance(items,Textitem) ) or  (isinstance(items, EllipseItem))):
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
            #~ self.delete = QtGui.QAction(self.tr('delete'), self)
            #~ self.connect(self.delete, QtCore.SIGNAL('triggered()'), self.deleteItem)
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
            for items in (qgraphicsitem for qgraphicsitem in self.rubberbandlist if (isinstance(qgraphicsitem,Textitem) or isinstance(qgraphicsitem, EllipseItem))):
                self.fitInView(self.startScenepos.x(),self.startScenepos.y(),self.rubberbandWidth,self.rubberbandHeight,Qt.Qt.KeepAspectRatio)
                if((self.matrix().m11()>=1.0)and(self.matrix().m22() >=1.0)):
                    for item in ( Txtitem for Txtitem in self.sceneContainerPt.items() if ( isinstance (Txtitem, Textitem)  or isinstance(Txtitem,EllipseItem))):
                        item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, False)
        else:
            self.rubberbandlist = self.sceneContainerPt.items(self.endScenepos.x(),self.endScenepos.y(),abs(self.rubberbandWidth),abs(self.rubberbandHeight), Qt.Qt.IntersectsItemShape)
            for unselectitem in self.rubberbandlist:
                if unselectitem.isSelected() == True:
                    unselectitem.setSelected(0)
            for items in (qgraphicsitem for qgraphicsitem in self.rubberbandlist if( isinstance(qgraphicsitem,Textitem) or isinstance(qgraphicsitem,Ellipseitem) )):
                self.fitInView(self.endScenepos.x(),self.endScenepos.y(),abs(self.rubberbandWidth),abs(self.rubberbandHeight),Qt.Qt.KeepAspectRatio)
                if((self.matrix().m11()>=1.0)and(self.matrix().m22() >=1.0)):
                    for item in ( Txtitem for Txtitem in self.sceneContainerPt.items() if isinstance (Txtitem, Textitem)):
                        item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, False)
        self.rubberBandactive = False

class Widgetvisibility(Exception):pass    
class KineticsWidget(QtGui.QWidget):
    def __init__(self,size,modelPath,parent=None):
        QtGui.QWidget.__init__(self,parent)
        self.border = 10
        #print "size",size
        hLayout = QtGui.QGridLayout(self)
        self.setLayout(hLayout)
        self.sceneContainer = QtGui.QGraphicsScene(self)
        self.sceneContainer.setBackgroundBrush(QtGui.QColor(230,220,219,120))
        self.view = GraphicalView(self.sceneContainer,self.border)
        cmptMol = {}
        self.setupCompt(modelPath,cmptMol)
        #for k,v in cmptMol.items(): print k,v

        srcdesConnection = {}
        zombieType = ['ZombieReac','ZombieEnz','ZombieMMenz','ZombieSumFunc']
        self.setupItem(modelPath,zombieType,srcdesConnection)
        #for k,v in srcdesConnection.items():      print k,v,'\n'

        #pickled the color map here and loading the file
        pkl_file = open(os.path.join(PATH_KKIT_COLORMAPS,'rainbow2.pkl'),'rb')
        picklecolorMap = pickle.load(pkl_file)
        
        self.lineItem_dict = {}
        self.object2line = {}
        mooseObjectCord = {}
        xr,yr = self.coordinates(modelPath,mooseObjectCord)
        xr = xr+2
        yr = yr+2
        #for k,v in mooseObjectCord.items(): print k,v
        #print xr,yr
        
        #This is check which version of kkit, b'cos anything below kkit8 didn't had xyz co-ordinates
        allZero = "True"
        for cmpt,itemlist in cmptMol.items():
            for  items in (items for items in itemlist if len(items) != 0):
                for item in items:
                    iteminfo = element(item.path+'/info')
                    x = float( iteminfo.getField('x') )
                    y = float( iteminfo.getField('y') )
                    if x != 0.0 or y != 0.0:
                        allZero = False
                        break

        if allZero:
            msgBox = QtGui.QMessageBox()
            msgBox.setText("The Layout module works for kkit version 8 or higher.")
            msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
            msgBox.exec_()
            raise Widgetvisibility()
        else:
            fnt = QtGui.QFont('Helvetica',11)
            self.qGraCompt = {}
            self.mooseId_GText = {}
            for cmpt,itemlist in cmptMol.items():
                self.createCompt(cmpt)
                comptRef = self.qGraCompt[cmpt]
                for  items in (items for items in itemlist if len(items) != 0):
                    for item in items:
                        bgcolor = ''
                        textcolor = ''
                        xx = '' 
                        yy = ''
                        xx, yy = mooseObjectCord[item.path]
                        x = float(xx)*xr
                        y = -float(yy)*yr
                        if((item.parent).class_ =='ZombieEnz'): 
                            textcolor = Annotator((item.parent).path+'info').getField('textColor')
                            bgcolor = Annotator((item.parent).path+'/info').getField('color')
                        else:
                            textcolor = Annotator(item.path+'info').getField('textColor')
                            bgcolor = Annotator(item.path+'/info').getField('color')
                        textcolor,bgcolor = self.colorCheck(item,textcolor,bgcolor,picklecolorMap)
                        if( (item.class_ == 'ZombiePool') and ((item.parent).class_ != 'ZombieEnz')): 
                            textcolor = Annotator(item.path+'/info').getField('textColor')
                            bgcolor = Annotator(item.path+'/info').getField('color')
                            textcolor,bgcolor = self.colorCheck(item,textcolor,bgcolor,picklecolorMap)
                            pItem = Textitem(comptRef,item)
                            pItem.setFont(fnt)
                            pItem.setPos(x,y)
                            pItem.setDefaultTextColor(QtGui.QColor(textcolor))
                            textbgcolor = "<html><body bgcolor='"+bgcolor+"'>"+item.name+"</body></html>"
                            pItem.setHtml(QtCore.QString(textbgcolor))
                            self.mooseId_GText[item.getId()] = pItem
                            self.connect(pItem, QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"), self.emitItemtoEditor)
                            pItem.positionChange.connect(self.positionChange)
                            pItem.selectedChange.connect(self.emitItemtoEditor)
        
                        elif( (item.parent).class_ != 'ZombieEnz'):
                            
                            textcolor = Annotator(item.path+'/info').getField('textColor')
                            bgcolor = 'black'
                            textcolor,bgcolor = self.colorCheck(item,textcolor,bgcolor,picklecolorMap)
                            if item.class_ == 'ZombieReac':
                                pItem = EllipseItem(x,y,5,5,comptRef,item)
                            else:
                                pItem = EllipseItem(x,y,15,15,comptRef,item)
                                pItem.setBrush(QtGui.QColor(textcolor))
                            self.mooseId_GText[item.getId()] = pItem
                            pItem.ellemitter.connect(pItem.ellemitter, QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.emitItemtoEditor)
                            pItem.ellemitter.connect(pItem.ellemitter,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.positionChange)
                            

                            '''
                        elif(item.class_ == 'ZombieReac'):
                            textcolor = Annotator(item.path+'/info').getField('textColor')
                            bgcolor = 'black'
                            textcolor,bgcolor = self.colorCheck(item,textcolor,bgcolor,picklecolorMap)
                            arrow = QtGui.QPolygonF()
                            r = 10
                            width = math.sin(math.radians(60))*r
                            height = math.cos(math.radians(60))*r
                            srcXArr1 = x + width
                            srcYArr1 = y + height
                            width = math.sin(math.radians(-120))*r
                            height = math.cos(math.radians(-120))*r
                            srcXArr = x+30 + width
                            srcYArr = y + height
                            arrow.append(QtCore.QPointF(x+30,y))
                            arrow.append(QtCore.QPointF(srcXArr,srcYArr))
                            arrow.append(QtCore.QPointF(x+30,y))
                            arrow.append(QtCore.QPointF(x,y))
                            arrow.append(QtCore.QPointF(srcXArr1,srcYArr1))
                            arrow.append(QtCore.QPointF(x,y))
                            pItem = polygonItem(arrow,comptRef,item)
                            #pItem.setBrush(QtGui.QColor(textcolor))
                            self.mooseId_GText[item.getId()] = pItem
                            #pItem.setPen(QtGui.QColor('blue'))
                            #pItem.setBrush(QtGui.QColor('blue'))
                            #pItem.setPen(QtGui.QPen(QtCore.Qt.black))
                            #self.sceneContainer.addItem(pItem)
                            #self.sceneContainer.addPolygon(arrow,QtGui.QPen(QtCore.Qt.black, 3, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
                            '''
                        else:
                            iteminfo = (item.parent).path+'/info'
                            textcolor = Annotator(iteminfo).getField('textColor')
                            bgcolor = 'black'
                            textcolor,bgcolor = self.colorCheck(item,textcolor,bgcolor,picklecolorMap)
                            print "X",iteminfo,item,textcolor   
                            pItem = QtGui.QGraphicsRectItem(x,y,15,15,comptRef)
                            pItem = RectCompt(self,x,y,15,15,self)
                            pItem.setBrush(QtGui.QColor(textcolor))
                            pItem.Rectemitter.connect(pItem.Rectemitter,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.positionChange)
                            pItem.Rectemitter.connect(pItem.Rectemitter,QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.emitItemtoEditor)
                            self.mooseId_GText[item.getId()] = pItem
                        
        for k, v in self.qGraCompt.items():
            rectcompt = v.childrenBoundingRect()
            v.setRect(rectcompt.x()-10,rectcompt.y()-10,(rectcompt.width()+20),(rectcompt.height()+20))
            v.setPen(QtGui.QPen(Qt.QColor(66,66,66,100),10,QtCore.Qt.SolidLine,QtCore.Qt.RoundCap,QtCore.Qt.RoundJoin ))
            v.Rectemitter.connect(v.Rectemitter,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.positionChange)
            v.Rectemitter.connect(v.Rectemitter,QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.emitItemtoEditor)

        for inn,out in srcdesConnection.items():
            if ((len(filter(lambda x:isinstance(x,list), out))) != 0):
                for items in (items for items in out[0] ):
                    src = ""
                    des = ""
                    src = self.mooseId_GText[inn.getId()]
                    des = self.mooseId_GText[items.getId()]
                    self.lineCord(src,des,inn,'no')
                for items in (items for items in out[1] ):
                    src = ""
                    des = ""
                    des = self.mooseId_GText[inn.getId()]
                    src = self.mooseId_GText[items.getId()]
                    self.lineCord(src,des,inn,'yes')
            else:
                for items in (items for items in out ):
                    src = ""
                    des = ""
                    src = self.mooseId_GText[inn.getId()]
                    des = self.mooseId_GText[items.getId()]
                    self.lineCord(src,des,inn,'yes')
        
        self.view.fitInView(self.sceneContainer.itemsBoundingRect())
        #self.view.fitInView(self.sceneContainer.sceneRect().x()-10,self.sceneContainer.sceneRect().y()-10,self.sceneContainer.sceneRect().width()+20,self.sceneContainer.sceneRect().height()+20,Qt.Qt.IgnoreAspectRatio)
        #self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
        hLayout.addWidget(self.view)
        #self.view.show()
    
    def coordinates (self,filePath,mobject_Cord):
        cPath = filePath+'/##[TYPE=MeshEntry]'
        for meshEnt in wildcardFind(cPath):
            x = []
            y = []
            molList = []
            reList = []
            for mitem in Neutral(meshEnt).getNeighbors('remesh'):
                if ( (mitem[0].class_ != 'GslIntegrator')):
                    if ((mitem[0].parent).class_ == 'ZombieEnz'):
                        xx = float(element((mitem[0].parent).path+'/info').getField('x'))
                        x.append(xx)
                        yy = float(element((mitem[0].parent).path+'/info').getField('y')-0.9)
                        y.append(yy)
                        mobject_Cord[mitem[0].path] = xx,yy
                    else:
                        xx1 = float(element(mitem.path+'/info').getField('x'))
                        x.append(xx1)
                        yy1 = float(element(mitem.path+'/info').getField('y'))
                        y.append(yy1)
                        mobject_Cord[mitem.path] = xx1,yy1
                            
            for reitem in Neutral(meshEnt).getNeighbors('remeshReacs'):
                xxr = float(element(reitem.path+'/info').getField('x'))
                x.append(xxr)
                yyr = float(element(reitem.path+'/info').getField('y'))
                y.append(yyr)
                mobject_Cord[reitem.path] = xxr,yyr
        xratio = ((max(x))-(min(x)))
        yratio = ((max(y))-(min(y)))
        return(xratio,yratio)
    
    #setting up compartments and its children    
    def setupCompt(self,modelPath,cmptmolDict):
        cPath = modelPath+'/##[TYPE=MeshEntry]'
        for meshEnt in wildcardFind(cPath):
            molList = []
            reList = []
            for mitem in Neutral(meshEnt).getNeighbors('remesh'):
                if ( (mitem[0].class_ != 'GslIntegrator')): # and ((mitem[0].parent).class_ != 'ZombieEnz') ):
                    molList.append(element(mitem))
            for reitem in Neutral(meshEnt).getNeighbors('remeshReacs'):
                reList.append(element(reitem))
            cmptmolDict[meshEnt] = molList,reList

    def setupItem(self,modlePath,searObject,cntDict):
        for zombieObj in searObject:
            path = modlePath+'/##[TYPE='+zombieObj+']'
            if zombieObj != 'ZombieSumFunc':
                for items in wildcardFind(path):
                    sublist = []
                    prdlist = []
                    for sub in items.getNeighbors('sub'): 
                        sublist.append(element(sub))
                    for prd in items.getNeighbors('prd'):
                        prdlist.append(element(prd))
                    if (zombieObj == 'ZombieEnz') :
                        for enzpar in items.getNeighbors('toZombieEnz'):
                            sublist.append(element(enzpar))
                        for cplx in items.getNeighbors('cplxDest'):
                            prdlist.append(element(cplx))
                    if (zombieObj == 'ZombieMMenz'):
                        for enzpar in items.getNeighbors('enzDest'):
                            sublist.append(element(enzpar))
                    cntDict[items] = sublist,prdlist
            else:
                #ZombieSumFunc adding inputs
                for items in wildcardFind(path):
                    inputlist = []
                    outputlist = []
                    funplist = []
                    nfunplist = []
                    for inpt in items.getNeighbors('input'):
                        inputlist.append(element(inpt))
                    for zfun in items.getNeighbors('output'): funplist.append(element(zfun))
                    for i in funplist: nfunplist.append(element(i).getId())
                    nfunplist = list(set(nfunplist))
                    if(len(nfunplist) > 1): print "SumFunPool has multiple Funpool"
                    else:
                        for el in funplist:
                            if(element(el).getId() == nfunplist[0]):
                                cntDict[element(el)] = inputlist
                                break
                            
    def colorCheck(self,item,textColor,bgcolor,pklcolor):
        if(textColor == ''): textColor = 'green'
        if(bgcolor == ''): bgcolor = 'blue'
        if(textColor == bgcolor): textColor = self.randomColor()
        hexchars = "0123456789ABCDEF"
        if(isinstance(textColor,(list,tuple))):
            r,g,b = textColor[0],textColor[1],textColor[2]
            textColor = "#"+ hexchars[r / 16] + hexchars[r % 16] + hexchars[g / 16] + hexchars[g % 16] + hexchars[b / 16] + hexchars[b % 16]
        elif ((not isinstance(textColor,(list,tuple)))):
            if textColor.isdigit():
                tc = int(textColor)
                tc = (tc * 2 )
                r,g,b = pklcolor[tc]
                textColor = "#"+ hexchars[r / 16] + hexchars[r % 16] + hexchars[g / 16] + hexchars[g % 16] + hexchars[b / 16] + hexchars[b % 16]
        if ((not isinstance(bgcolor,(list,tuple)))):
            if bgcolor.isdigit():
                tc = int(bgcolor)
                tc = (tc * 2 )
                r,g,b = pklcolor[tc]
                bgcolor = "#"+hexchars[r/16] + hexchars[r % 16] + hexchars[g / 16] + hexchars[g % 16] + hexchars[b / 16] + hexchars[b % 16]
        return(textColor,bgcolor)
   
    def randomColor(self):
        red = int(random.uniform(0, 255))
        green = int(random.uniform(0, 255))
        blue = int(random.uniform(0, 255))
        return (red,green,blue)
    
    def createCompt(self,key):
        self.new_Compt = RectCompt(self,0,0,0,0,key)
        self.qGraCompt[key] = self.new_Compt
        self.new_Compt.setRect(10,10,10,10)
        self.sceneContainer.addItem(self.new_Compt)
    
    def emitItemtoEditor(self,mooseObject):
        if(isinstance(mooseObject,Textitem)):
            #need to pass mooseObject for objectoreditor
           mooseObject = element(next((k for k,v in self.mooseId_GText.items() if v == mooseObject), None))

        self.emit(QtCore.SIGNAL("itemDoubleClicked(PyQt_PyObject)"), mooseObject)
        
    def positionChange(self,mooseObject):
        #If the item position changes, the corresponding arrow's are claculated
        
        if( (isinstance(mooseObject, Textitem)) and (isinstance(mooseObject, EllipseItem))):
            self.updatearrow(mooseObject)
            for k, v in self.qGraCompt.items():
                rectcompt = v.childrenBoundingRect()
                v.setRect(rectcompt.x()-10,rectcompt.y()-10,(rectcompt.width()+20),(rectcompt.height()+20))
        else:
            for k, v in self.qGraCompt.items():
                for rectChilditem in v.childItems():
                    self.updatearrow(rectChilditem)

    def lineCord(self,src,des,source,drawarrowhead):
        
        if( (src == "") & (des == "") ):
            print "Source or destination is missing or incorrect"
        else:
            srcdes_list= [src,des]
            arrow = self.calArrow(src,des,drawarrowhead)
            if(source.class_ == "ZombieReac"):
                qgLineitem = self.sceneContainer.addPolygon(arrow,QtGui.QPen(QtCore.Qt.green, 1, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
            elif( (source.class_ == "ZombieEnz") or (source.class_ == "ZombieMMenz")):
                qgLineitem = self.sceneContainer.addPolygon(arrow,QtGui.QPen(QtCore.Qt.red, 1, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
            elif( (source.class_ == "ZombiePool") or (source.class_ == "ZombieFuncPool")):
                qgLineitem = self.sceneContainer.addPolygon(arrow,QtGui.QPen(QtCore.Qt.blue, 1, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
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
      
    def calArrow(self,src,des,drawarrowhead):
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
        #if drawarrowhead == 'yes':
        srcRect = QtCore.QRectF(sX,sY,sw,sh)
        desRect = QtCore.QRectF(dX,dY,dw,dh)
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
            arrow.append(QtCore.QPointF(lineSrcpoint.x(),lineSrcpoint.y()))
            arrow.append(QtCore.QPointF(lineDespoint.x(),lineDespoint.y()))
            #Arrow head is drawned if the distance between src and des line is >8 just for clean appeareance
            if(abs(lineSrcpoint.x()-lineDespoint.x()) > 8 or abs(lineSrcpoint.y()-lineDespoint.y())>8):
               #Arrow head for Source is calculated
                if drawarrowhead != 'no':
                    degree = 60
                    srcXArr1,srcYArr1= self.arrowHead(srcAngle,degree,lineSrcpoint)
                    degree = 120
                    srcXArr2,srcYArr2 = self.arrowHead(srcAngle,degree,lineSrcpoint)
                    arrow.append(QtCore.QPointF(lineSrcpoint.x(),lineSrcpoint.y()))
                    arrow.append(QtCore.QPointF(srcXArr1,srcYArr1))
                    arrow.append(QtCore.QPointF(lineSrcpoint.x(),lineSrcpoint.y()))
                    arrow.append(QtCore.QPointF(srcXArr2,srcYArr2))
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
            #  Added & substracted 5 for x and y making a outerboundary for the item for drawing
            lItemSP = QtCore.QLineF(X,Y,X+w,Y)
            boundintersect= lItemSP.intersect(centerLine,linePoint)
            if (boundintersect == 1):
                return centerLine.angle()
            else:
                lItemSP = QtCore.QLineF(X+w,Y,X+w,Y+h)
                boundintersect= lItemSP.intersect(centerLine,linePoint)
                if (boundintersect == 1):
                    return centerLine.angle()
                else:
                    lItemSP = QtCore.QLineF(X+w,Y+h,X,Y+h)
                    boundintersect= lItemSP.intersect(centerLine,linePoint)
                    if (boundintersect == 1):
                        return centerLine.angle()
                    else:
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
    
    def updateItemSlot(self, mooseObject):
        #In this case if the name is updated from the keyboard both in mooseobj and gui gets updation
        changedItem = ''
        for changedItem in (item for item in self.sceneContainer.items() if isinstance(item, Textitem) and mooseObject.getId() == item.mooseObj_.getId()):
            break
        
        if isinstance(changedItem,Textitem):
            changedItem.updateSlot()
            self.positionChange(changedItem.mooseObj_)
    
    def updatearrow(self,qGTextitem):
        listItem = self.object2line[qGTextitem]
        for ql, va in listItem:
            srcdes = self.lineItem_dict[ql]
            #print "!!!!",srcdes[0],srcdes[1],"\n",ql,va
            if (isinstance(srcdes[1],Textitem) and isinstance(srcdes[0],EllipseItem)): 
                arrow = self.calArrow(srcdes[0],srcdes[1],'no')
            if (isinstance(srcdes[0],Textitem) and isinstance(srcdes[1],EllipseItem)): 
                arrow = self.calArrow(srcdes[0],srcdes[1],'yes')
            
            ql.setPolygon(arrow)
    
    def keyPressEvent(self,event):
        key = event.key()
        if key == QtCore.Qt.Key_A:
            self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
            #self.view.fitInView(self.sceneContainer.sceneRect().x()-10,self.sceneContainer.sceneRect().y()-10,self.sceneContainer.sceneRect().width()+20,self.sceneContainer.sceneRect().height()+20,Qt.Qt.IgnoreAspectRatio)
        elif (key == 46 or key == 62):
            self.view.scale(1.1,1.1)
        elif (key == 44 or key == 60):
            self.view.scale(1/1.1,1/1.1)
    
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    size = QtCore.QSize(1800,1600)
    modelPath = 'Kholodenko'
    #modelPath = 'enz_class_exp'
    #modelPath = 'reaction'
    #modelPath = 'Osc_cspace_ref'
    #modelPath = 'traff_nn_diff_TRI'
    #modelPath = 'acc68'
    
    loadModel('/home/harsha/dh_branch/Demos/Genesis_files/china_course/'+modelPath+'.g','/'+modelPath)
    dt = KineticsWidget(size,'/'+modelPath)
    dt.show()
    sys.exit(app.exec_())
