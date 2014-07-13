from PyQt4 import QtCore, QtGui,Qt
import sys
import config
from modelBuild import *

class GraphicalView(QtGui.QGraphicsView):
    def __init__(self,editorWidgetBase,modelRoot,parent,border,layoutPt,createdItem):
        QtGui.QGraphicsView.__init__(self,parent)
        self.setScene(parent)
        self.editorWigetBaseref = editorWidgetBase
        self.modelRoot = modelRoot
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
        self.setAcceptDrops(True)
        self.createdItem = createdItem
        # All the object which are stacked on the scene are listed
        self.stackOrder = self.sceneContainerPt.items(Qt.Qt.DescendingOrder)
        #From stackOrder selecting only compartment
        self.cmptStackorder = [i for i in self.stackOrder if isinstance(i,ComptItem)]
        
    def mousePressEvent(self, event):
        selectedItem = None
        if event.buttons() == QtCore.Qt.LeftButton:
            self.startingPos = event.pos()
            self.startScenepos = self.mapToScene(self.startingPos)
            self.deviceTransform = self.viewportTransform()
            viewItem = self.items(self.startingPos)
            #All previously seted zValue made zero
            for i in self.cmptStackorder:
                i.setZValue(0)
            itemIndex = 0
            kkitItem  = [j for j in viewItem if isinstance(j,KineticsDisplayItem)]
            comptItem = [k for k in viewItem if isinstance(k,ComptItem)]
            if kkitItem:
                for displayitem in kkitItem:
                    ''' mooseItem(child) ->compartment (parent) But for cplx
                        cplx(child)->Enz(parent)->compartment(super parent)
                        to get compartment for cplx one has to get super parent
                    '''
                    if isinstance(displayitem,CplxItem):
                        displayitem = displayitem.parentItem()
                    itemIndex = self.cmptStackorder.index(displayitem.parentItem())
                    selectedItem = displayitem
                    displayitem.parentItem().setZValue(1)
            elif not kkitItem and comptItem:
                for cmpt in comptItem:
                    for previouslySelected in self.sceneContainerPt.selectedItems():
                        if previouslySelected.isSelected() == True:
                            previouslySelected.setSelected(False)
                    xs = cmpt.mapToScene(cmpt.boundingRect().topLeft()).x()+self.border/2
                    ys = cmpt.mapToScene(cmpt.boundingRect().topLeft()).y()+self.border/2
                    xe = cmpt.mapToScene(cmpt.boundingRect().bottomRight()).x()-self.border/2
                    ye = cmpt.mapToScene(cmpt.boundingRect().bottomRight()).y()-self.border/2
                    xp = self.startScenepos.x()
                    yp = self.startScenepos.y()
                    #if on border rubberband is not started, but called parent class for default implementation
                    if(  ((xp > xs-self.border/2) and (xp < xs+self.border/2) and (yp > ys-self.border/2) and (yp < ye+self.border/2) )or
                         ((xp > xs+self.border/2) and (xp < xe-self.border/2) and (yp > ye-self.border/2) and (yp < ye+self.border/2) ) or 
                         ((xp > xs+self.border/2) and (xp < xe-self.border/2) and (yp > ys-self.border/2) and (yp < ys+self.border/2) ) or 
                         ((xp > xe-self.border/2) and (xp < xe+self.border/2) and (yp > ys-self.border/2) and (yp < ye+self.border/2) ) ):
                        if self.cmptStackorder:
                            itemIndex = self.cmptStackorder.index(cmpt)
                        
                        cmpt.setZValue(1)
                        selectedItem = cmpt
                        break

            if selectedItem == None:
                #if mousepressed is not on any kineticsDisplayitem or on compartment border, 
                #then rubberband is made active
                enableRubberband = False
                self.customrubberBand = QtGui.QRubberBand(QtGui.QRubberBand.Rectangle,self)
                self.customrubberBand.setGeometry(QtCore.QRect(self.startingPos,QtCore.QSize()))
                self.customrubberBand.show()
            else:
                QtGui.QGraphicsView.mousePressEvent(self,event)

    def mouseMoveEvent(self,event):
        QtGui.QGraphicsView.mouseMoveEvent(self, event)
        if( (self.customrubberBand) and (event.buttons() == QtCore.Qt.LeftButton)):
            self.endingPos = event.pos()
            self.endingScenepos = self.mapToScene(self.endingPos)
            self.rubberbandWidth = self.endingScenepos.x()-self.startScenepos.x()
            self.rubberbandHeight = self.endingScenepos.y()-self.startScenepos.y()
            self.customrubberBand.setGeometry(QtCore.QRect(self.startingPos, event.pos()).normalized())
            #unselecting any previosly selected item in scene
            for preSelectedItem in self.sceneContainerPt.selectedItems():
                preSelectedItem.setSelected(False)
            #since it custom rubberband I am checking if with in the selected area any textitem, if s then setselected to true
            rbandSelection = self.sceneContainerPt.items(self.startScenepos.x(),self.startScenepos.y(),self.rubberbandWidth,self.rubberbandHeight,Qt.Qt.IntersectsItemShape)
            for item in rbandSelection:
                if isinstance(item,KineticsDisplayItem) and item.isSelected() == False:
                        item.setSelected(True)
                        
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
            for items in (qgraphicsitem for qgraphicsitem in self.rubberbandlist if isinstance(qgraphicsitem,PoolItem)):
                self.fitInView(self.startScenepos.x(),self.startScenepos.y(),self.rubberbandWidth,self.rubberbandHeight,Qt.Qt.KeepAspectRatio)
                if((self.matrix().m11()>=1.0)and(self.matrix().m22() >=1.0)):
                    for item in ( Txtitem for Txtitem in self.sceneContainerPt.items() if isinstance(Txtitem,PoolItem) ):
                        item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, False)
        else:
            self.rubberbandlist = self.sceneContainerPt.items(self.endScenepos.x(),self.endScenepos.y(),abs(self.rubberbandWidth),abs(self.rubberbandHeight), Qt.Qt.IntersectsItemShape)
            for unselectitem in self.rubberbandlist:
                if unselectitem.isSelected() == True:
                    unselectitem.setSelected(0)
            for items in (qgraphicsitem for qgraphicsitem in self.rubberbandlist if isinstance(qgraphicsitem,PoolItem)):
                self.fitInView(self.endScenepos.x(),self.endScenepos.y(),abs(self.rubberbandWidth),abs(self.rubberbandHeight),Qt.Qt.KeepAspectRatio)
                if((self.matrix().m11()>=1.0)and(self.matrix().m22() >=1.0)):
                    for item in ( Txtitem for Txtitem in self.sceneContainerPt.items() if isinstance (Txtitem, PoolItem)):
                        item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, False)
        self.rubberBandactive = False

        
    def resizeEvent1(self, event):
        """ zoom when resize! """
        self.fitInView(self.sceneContainerPt.itemsBoundingRect().x()-10,self.sceneContainerPt.itemsBoundingRect().y()-10,self.sceneContainerPt.itemsBoundingRect().width()+20,self.sceneContainerPt.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
        QtGui.QGraphicsView.resizeEvent(self, event)
        
    def wheelEvent(self,event):
        factor = 1.41 ** (event.delta() / 240.0)
        self.scale(factor, factor)
        

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat('text/plain'):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat('text/plain'):
            event.acceptProposedAction()

    def eventFilter(self, source, event):
        if (event.type() == QtCore.QEvent.Drop):
            print "dropEvent has happened"

    def dropEvent(self, event):
        """Insert an element of the specified class in drop location"""
        if not event.mimeData().hasFormat('text/plain'):
            return
        pos = event.pos()
        viewItems = self.items(pos)
        mapToscene = self.mapToScene(event.pos())
        newString = str(event.mimeData().text())
        Item = NewObject(self.editorWigetBaseref,self,self.modelRoot,newString,mapToscene,self.createdItem)
        self.sceneContainerPt.addItem(Item)
        Item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations,True)
        self.setScene(self.sceneContainerPt)
        event.accept()
          
