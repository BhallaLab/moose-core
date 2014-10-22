from PyQt4 import QtCore, QtGui,Qt
import sys
import config
from modelBuild import *
from constants import *
from PyQt4.QtGui import QPixmap
from PyQt4.QtGui import QImage
from PyQt4.QtGui import QGraphicsPixmapItem
from kkitCalcArrow import *
from kkitOrdinateUtil import *

class GraphicalView(QtGui.QGraphicsView):
    def __init__(self, modelRoot,parent,border,layoutPt,createdItem):
        QtGui.QGraphicsView.__init__(self,parent)
        self.state = { "press"  :   { "mode"    : INVALID
                                    , "item"    : None
                                    , "sign"    : None
                                    }
                     , "release"    :   { "mode"    : INVALID
                                        , "item"    : None
                                        , "sign"    : None
                                        }
                     }
        self.connectionSignImagePath = "../gui/icons/connection.png"
        self.connectionSignImage     = QImage(self.connectionSignImagePath)
        self.connectionSign          = None

        self.setScene(parent)
        self.modelRoot = modelRoot
        self.sceneContainerPt = parent
        self.setDragMode(QtGui.QGraphicsView.RubberBandDrag)
        self.itemSelected = False
        self.customrubberBand=0
        self.rubberbandWidth = 0
        self.rubberbandHeight = 0
        self.moved = False
        self.showpopupmenu = False
        self.popupmenu4rlines = True
        self.border = 6
        self.setRenderHints(QtGui.QPainter.Antialiasing)
        self.layoutPt = layoutPt
        #self.setAcceptDrops(True)
        #self.createdItem = createdItem
        # All the object which are stacked on the scene are listed
        self.stackOrder = self.sceneContainerPt.items(Qt.Qt.DescendingOrder)
        #From stackOrder selecting only compartment
        self.cmptStackorder = [i for i in self.stackOrder if isinstance(i,ComptItem)]
        self.viewBaseType = " "
        self.object2line = defaultdict(list)
        self.subsetObject2line = {}
        #self.qGraCompt = {}
        #self.mooseId_GObj = {}
        self.c= 1
        self.p = 1
    def setobject2line(self,path):
        self.object2line = path
        
    def setRefWidget(self,path):
        self.viewBaseType = path
    
    
    def mousePressEvent(self, event):
        #print "---------------\n mousePressEvent",self.itemAt(event.pos()),event.pos(), " mtos",self.mapToScene(event.pos())
        print "here at mousepressEvent ",self.sceneContainerPt.items()
        selectedItem = None
        if self.viewBaseType == "editorView":
            if event.buttons() == QtCore.Qt.LeftButton:
                self.startingPos = event.pos()
                self.startScenepos = self.mapToScene(self.startingPos)
                self.deviceTransform = self.viewportTransform()
                viewItem = self.items(self.startingPos)
                print "viewItem ",viewItem

                #All previously seted zValue made zero
                for i in self.cmptStackorder:
                    i.setZValue(0)
                itemIndex = 0
                arrowItem = [j for j in viewItem if isinstance(j,QtGui.QGraphicsPolygonItem)]
                kkitItem  = [j for j in viewItem if isinstance(j,KineticsDisplayItem)]
                comptItem = [k for k in viewItem if isinstance(k,ComptItem)]
                print "ComptItem ",comptItem, "kkitItem",kkitItem
                connectionSign = [k for k in viewItem if isinstance(k, QGraphicsPixmapItem)]
                print  "connectionSign ",connectionSign
                if len(connectionSign) != 0:
                    sign = connectionSign[0]
                    self.state["press"]["mode"] = CONNECTION
                    self.state["press"]["sign"] = self.connectionSign
                    print "$$@ \t \t ",self.state["press"]["mode"], "1 ",self.state["press"]["sign"],"2",self.state["press"]["item"]
                    self.startingPos = event.pos()
                    self.startScenepos = self.mapToScene(self.startingPos)
                    qgrapItem = self.state["press"]["item"]
                    print "QGraphicsItem ",qgrapItem
                    print "bR ",qgrapItem.sceneBoundingRect()
                    srcdes = [qgrapItem,'',"s",1]
                    foo = qgrapItem.sceneBoundingRect()
                    #self.startScenepos=fd.sceneBoundingRect()
                    #arrow = calcArrow(srcdes,False,1)
                    # print "arror ---",arrow
                    self.qlineItem = QtGui.QGraphicsLineItem(foo.x(),foo.y(),foo.x(),foo.y(),None)
                    self.sceneContainerPt.addItem(self.qlineItem)
                    return
                if arrowItem:
                    for k,tupValue in self.object2line.iteritems():
                        for l,v in enumerate(tupValue):
                            pass
                            #print "can delete QGraphicsPolygonItem"
                            # if v[0] == arrowItem[0]:
                            #     print "--",k,v[1]
                            #     if self.popupmenu4rlines:
                            #         popupmenu = QtGui.QMenu('PopupMenu', self)
                            #         self.delete = QtGui.QAction(self.tr('delete'), self)
                            #         self.connect(self.delete, QtCore.SIGNAL('triggered()'), self.testItem)
                            #         popupmenu.addAction(self.delete)
                            #         popupmenu.exec_(event.globalPos())
                            #     self.popupmenu4rlines = False
                if kkitItem:
                    for displayitem in kkitItem:
                        ''' mooseItem(child) ->compartment (parent) But for cplx
                            cplx(child)->Enz(parent)->compartment(super parent)
                            to get compartment for cplx one has to get super parent
                        '''
                        if isinstance(displayitem,CplxItem):
                            displayitem = displayitem.parentItem()
                        #itemIndex = self.cmptStackorder.index(displayitem.parentItem())
                        selectedItem = displayitem
                        displayitem.parentItem().setZValue(1)
                        print "\n  \\\\\\\\\\\\\\\\\\\\ \n"
                        if isinstance(displayitem,PoolItem) or isinstance(displayitem,FuncItem):
                            rectangle = displayitem.bg.rect()
                            print "rectangle",rectangle
                            
                        else:
                            srcobj = displayitem.gobj
                            rectangle = srcobj.boundingRect()

                        if self.connectionSign is not None:
                            self.sceneContainerPt.removeItem(self.connectionSign)
                            self.connectionSign = None
                        self.state["press"]["item"] = displayitem    
                        self.connectionSign =  QGraphicsPixmapItem(
                            QPixmap.fromImage(self.connectionSignImage.scaled( rectangle.height()
                                                                        , rectangle.height() 
                                                                        )))
                        self.connectionSign.setPos(0.0,0.0)
                        position = displayitem.mapToParent(rectangle.topRight())
                        self.connectionSign.moveBy(position.x(), position.y() - rectangle.height() / 2.0)
                        self.sceneContainerPt.addItem(self.connectionSign)


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

        elif self.viewBaseType == "runView":
            pos = event.pos()
            item = self.itemAt(pos)
            itemClass = type(item).__name__
            if ( itemClass!='ComptItem' and itemClass != 'QGraphicsPolygonItem'):
                self.setCursor(Qt.Qt.CrossCursor)
                mimeData = QtCore.QMimeData()
                mimeData.setText(item.mobj.name)
                mimeData.setData("text/plain", "")
                mimeData.data =(self.modelRoot,item.mobj)
                drag = QtGui.QDrag(self)
                drag.setMimeData(mimeData)
                dropAction = drag.start(QtCore.Qt.MoveAction)
                self.setCursor(Qt.Qt.ArrowCursor)

    
    def mouseMoveEvent(self,event):
        if self.connectionSign is not None:
            if self.state["press"]["mode"] == CONNECTION:
                destination = self.mapToScene(event.pos())
                #self.state["press"]["sign"].setPos(destination)
                return

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
        if self.state["press"]["mode"] == CONNECTION:
            desPos =self.mapToScene(event.pos())
            destination = self.items(event.pos())
            src = self.state["press"]["item"]
            des  = [j for j in destination if isinstance(j,KineticsDisplayItem)]
            if len(des):
                self.populate_srcdes(src.mobj,des[0].mobj)

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
                #print "selecteditems ",selecteditems
                if self.rubberbandWidth != 0 and self.rubberbandHeight != 0 and len(selecteditems) != 0 :
                    self.showpopupmenu = True
        self.itemSelected = False
        if self.showpopupmenu:
            popupmenu = QtGui.QMenu('PopupMenu', self)
            self.delete = QtGui.QAction(self.tr('delete'), self)
            self.connect(self.delete, QtCore.SIGNAL('triggered()'), self.deleteItem)
            self.zoom = QtGui.QAction(self.tr('zoom'), self)
            self.connect(self.zoom, QtCore.SIGNAL('triggered()'), self.zoomItem)
            self.move = QtGui.QAction(self.tr('move'), self)
            self.connect(self.move, QtCore.SIGNAL('triggered()'), self.moveItem)
            popupmenu.addAction(self.delete)
            popupmenu.addAction(self.zoom)
            popupmenu.addAction(self.move)
            popupmenu.exec_(event.globalPos())
        self.showpopupmenu = False
    def moveItem(self):
      self.setCursor(Qt.Qt.CrossCursor)

    def deleteItem(self):
        vTransform = self.viewportTransform()
        if( self.rubberbandWidth > 0  and self.rubberbandHeight >0):
            self.rubberbandlist = self.sceneContainerPt.items(self.startScenepos.x(),self.startScenepos.y(),self.rubberbandWidth,self.rubberbandHeight, Qt.Qt.IntersectsItemShape)
            for unselectitem in self.rubberbandlist:
                if unselectitem.isSelected() == True:
                    unselectitem.setSelected(0)
            for items in (qgraphicsitem for qgraphicsitem in self.rubberbandlist):
                print "items in delete function",items
        else:
            self.rubberbandlist = self.sceneContainerPt.items(self.endScenepos.x(),self.endScenepos.y(),abs(self.rubberbandWidth),abs(self.rubberbandHeight), Qt.Qt.IntersectsItemShape)
            for unselectitem in self.rubberbandlist:
                if unselectitem.isSelected() == True:
                    unselectitem.setSelected(0)
            for items in (qgraphicsitem for qgraphicsitem in self.rubberbandlist):
                print "items in delete fnction -ve ",items
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
        self.view.fitInView(self.sceneContainer.itemsBoundingRect().x()-10,self.sceneContainer.itemsBoundingRect().y()-10,self.sceneContainer.itemsBoundingRect().width()+20,self.sceneContainer.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
        #self.fitInView(self.sceneContainerPt.itemsBoundingRect().x()-10,self.sceneContainerPt.itemsBoundingRect().y()-10,self.sceneContainerPt.itemsBoundingRect().width()+20,self.sceneContainerPt.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
        QtGui.QGraphicsView.resizeEvent(self, event)

    def wheelEvent(self,event):
        factor = 1.41 ** (event.delta() / 240.0)
        self.scale(factor, factor)


    def dragEnterEvent(self, event):
        if self.viewBaseType == "editorView":
            if event.mimeData().hasFormat('text/plain'):
                event.acceptProposedAction()
        else:
            pass

    def dragMoveEvent(self, event):
        if self.viewBaseType == "editorView":
            if event.mimeData().hasFormat('text/plain'):
                event.acceptProposedAction()
        else:
            pass

    def eventFilter(self, source, event):
        if self.viewBase == "editorView":
            if (event.type() == QtCore.QEvent.Drop):
                pass
        else:
            pass

    def dropEvent(self, event):
        """Insert an element of the specified class in drop location"""
        """ Pool and reaction should have compartment as parent, dropping outside the compartment is not allowed """
        """ Enz should be droped on the PoolItem which inturn will be under compartment"""
        if self.viewBaseType == "editorView":
            if not event.mimeData().hasFormat('text/plain'):
                return
            event_pos = event.pos()
            string = str(event.mimeData().text())
            createObj(self.viewBaseType,self,self.modelRoot,string,event_pos,self.layoutPt)

    def populate_srcdes(self,src,des):
        self.modelRoot = self.layoutPt.modelRoot
        callsetupItem = True
        if ( isinstance(moose.element(src),PoolBase) and ( (isinstance(moose.element(des),ReacBase) ) or isinstance(moose.element(des),EnzBase) )):
            moose.connect(src, 'reac', des, 'sub', 'OneToOne')
        elif(isinstance (moose.element(src),PoolBase) and (isinstance(moose.element(des),Function))):

            pFconnection = moose.connect( src, 'nOut', des.x[0], 'input' )
            print "here",pFconnection
        elif( isinstance(moose.element(src),Function) and (moose.element(des).className=="Pool") ):
                moose.connect(src, 'valueOut', des, 'increment', 'OneToOne')
        elif( isinstance(moose.element(src),Function) and (moose.element(des).className=="BufPool") ):
                moose.connect(src, 'valueOut', des, 'setConcInit', 'OneToOne')
        elif( isinstance(moose.element(src),Function) and (isinstance(moose.element(des),ReacBase) ) ):
                moose.connect(src, 'valueOut', des, 'numkf', 'OneToOne')
        elif( isinstance(moose.element(src),ReacBase) and (isinstance(moose.element(des),PoolBase) ) ):
            moose.connect(src, 'prd', des, 'reac', 'OneToOne')
        elif( isinstance(moose.element(src),EnzBase) and (isinstance(moose.element(des),PoolBase) ) ):
            moose.connect(src, 'prd', des, 'reac', 'OneToOne')
        elif( isinstance(moose.element(src),StimulusTable) and (isinstance(moose.element(des),PoolBase) ) ):
            moose.connect(src, 'output', des, 'setconcInit', 'OneToOne')
        else:
            srcString = moose.element(src).className
            desString = moose.element(des).className
            srcdesString = srcString+'--'+desString
            # QtGui.QMessageBox.information(None,'Connection Not possible','\'{srcString}\' Not allowed to connect \'{desString}\' '.format(srcString = srcString, desString=desString),QtGui.QMessageBox.Ok)
            QtGui.QMessageBox.information(None,'Connection Not possible','\'{srcdesString}\' not allowed to connect'.format(srcdesString = srcdesString),QtGui.QMessageBox.Ok)
            callsetupItem = False
            
        if callsetupItem:
            print " here in setupItem"
            setupItem(self.modelRoot,self.layoutPt.srcdesConnection)
            self.layoutPt.drawLine_arrow(False)
