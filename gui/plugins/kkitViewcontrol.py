from PyQt4 import QtCore, QtGui,Qt
import sys
import config
from modelBuild import *
from constants import *
from PyQt4.QtGui import QPixmap
from PyQt4.QtGui import QImage
from PyQt4.QtGui import QGraphicsPixmapItem
from PyQt4.QtGui import QGraphicsLineItem
from PyQt4.QtGui import QPen
from kkitCalcArrow import *
from kkitOrdinateUtil import *
import pprint

class GraphicalView(QtGui.QGraphicsView):
    def __init__(self, modelRoot,parent,border,layoutPt,createdItem):
        QtGui.QGraphicsView.__init__(self,parent)
        self.state = None
        self.resetState()
        self.connectionSign          = None
        self.connectionSource        = None
        self.expectedConnection      = None
        self.selections              = []
        self.connector               = None
        self.connectionSignImagePath = "../gui/icons/connection.png"
        self.connectionSignImage     = QImage(self.connectionSignImagePath)

        # self.expectedConnectionPen   = QPen()
        self.setScene(parent)
        self.modelRoot = modelRoot
        self.sceneContainerPt = parent
        self.setDragMode(QtGui.QGraphicsView.RubberBandDrag)
        self.itemSelected = False
        self.customrubberBand = None
        self.rubberbandWidth = 0
        self.rubberbandHeight = 0
        self.moved = False
        self.showpopupmenu = False
        self.popupmenu4rlines = True
        self.border = 6
        self.setRenderHints(QtGui.QPainter.Antialiasing)
        self.layoutPt = layoutPt
        # All the object which are stacked on the scene are listed
        self.stackOrder = self.sceneContainerPt.items(Qt.Qt.DescendingOrder)
        #From stackOrder selecting only compartment
        self.cmptStackorder = [i for i in self.stackOrder if isinstance(i,ComptItem)]
        self.viewBaseType = " "
      
    def setRefWidget(self,path):
        self.viewBaseType = path

    def resolveCompartmentInteriorAndBoundary(self, item, position):
        bound = item.rect().adjusted(3,3,-3,-3)
        return COMPARTMENT_INTERIOR if bound.contains(item.mapFromScene(position)) else COMPARTMENT_BOUNDARY

    def resetState(self):
        self.state = { "press"      :   { "mode"     :  INVALID
                                        , "item"     :  None
                                        , "sign"     :  None
                                        , "pos"    :  None
                                        }
                , "move"       :   { "happened": False
                                   }
                , "release"    :   { "mode"    : INVALID
                                   , "item"    : None
                                   , "sign"    : None
                                   }
                }


    def resolveItem(self, items, position):
        solution = None
        for item in items:
            if hasattr(item, "name"):
                #print(item.name)
                if item.name == ITEM:
                    return (item, ITEM)
                if item.name == COMPARTMENT:
                    solution = (item, self.resolveCompartmentInteriorAndBoundary(item, position))

        for item in items:
            if isinstance(item, QtGui.QGraphicsPixmapItem):
                return (item, CONNECTOR)
            if isinstance(item, QtGui.QGraphicsPolygonItem):
                return (item, CONNECTION)

        if solution is None:
            return (None, EMPTY)
        return solution

    def editorMousePressEvent(self, event):
        # self.deselectSelections()
        # if self.state["press"]["item"] is not None:
        #     self.state["press"]["item"].setSelected(False)
        # self.resetState()
        if event.buttons() == QtCore.Qt.LeftButton:
            self.clickPosition  = self.mapToScene(event.pos())
            (item, itemType) = self.resolveItem(self.items(event.pos()), self.clickPosition)
            self.state["press"]["mode"] = VALID
            self.state["press"]["item"] = item
            self.state["press"]["type"] = itemType
            self.state["press"]["pos"]  = event.pos()
            # self.layoutPt.plugin.mainWindow.objectEditSlot(self.state["press"]["item"].mobj, False)
        else:
            self.resetState()
        #print(self.state)


    def editorMouseMoveEvent(self, event):
        if self.state["press"]["mode"] == INVALID:
            self.state["move"]["happened"] = False
            return

        self.state["move"]["happened"] = True
        itemType = self.state["press"]["type"]
        item     = self.state["press"]["item"]

        if itemType == CONNECTOR:
            self.drawExpectedConnection(event)
            #print("connect objects")

        if itemType == ITEM:
            parent  = item.getParentGraphicsObject()
            initial = parent.mapFromScene(self.mapToScene(self.state["press"]["pos"]))
            final   = parent.mapFromScene(self.mapToScene(event.pos()))
            displacement = final - initial
            #print("Displacement", displacement),item.__class__, " par ",parent
            if not isinstance(item,FuncItem) and not isinstance(item,CplxItem):
                #print "item at displacement ",item
                item.moveBy(displacement.x(), displacement.y())
                if isinstance(item,PoolItem):
                    for funcItem in item.childItems():
                        if isinstance(funcItem,FuncItem):
                            self.layoutPt.updateArrow(funcItem)
            self.state["press"]["pos"] = event.pos()

            # item.setPos(item.getParentGraphicsObject().mapFromScene(self.mapToScene(event.pos())))

        if itemType == COMPARTMENT_BOUNDARY:
            initial = self.mapToScene(self.state["press"]["pos"])
            final = self.mapToScene(event.pos())
            displacement = final - initial
            #print("Displacement", displacement)
            item.moveBy(displacement.x(), displacement.y())            
            self.state["press"]["pos"] = event.pos()
            # QtGui.QGraphicsView.mouseMoveEvent(self, event)
            # self.state["press"]["item"].qgtextPositionChange.emit(self.state["press"]["item"].mobj, event.pos())
            # event.ignore()

        if itemType == COMPARTMENT_INTERIOR:
            if self.customrubberBand == None:
                self.customrubberBand = QtGui.QRubberBand(QtGui.QRubberBand.Rectangle,self)
                self.customrubberBand.show()

            startingPosition = self.state["press"]["pos"]
            endingPosition = event.pos()
            displacement   = endingPosition - startingPosition

            x0 = startingPosition.x() 
            x1 = endingPosition.x()
            y0 = startingPosition.y() 
            y1 = endingPosition.y()

            if displacement.x() < 0 :
                x0,x1= x1,x0

            if displacement.y() < 0 :
                y0,y1= y1,y0

            self.customrubberBand.setGeometry(QtCore.QRect(QtCore.QPoint(x0, y0), QtCore.QSize(abs(displacement.x()), abs(displacement.y()))))
            

        # if itemType == COMPARTMENT:
        #     rubberband selection

        # if itemType == COMPARTMENT_BOUNDARY:
            
        # if itemType == ITEM:
        #     dragging the item

    def drawExpectedConnection(self, event):
        sourcePoint      = self.connectionSource.mapToScene(
            self.connectionSource.boundingRect().center()
                                          )
        destinationPoint = self.mapToScene(event.pos())
        if self.expectedConnection is None:
            self.expectedConnection = QGraphicsLineItem( sourcePoint.x()
                                                       , sourcePoint.y()
                                                       , destinationPoint.x()
                                                       , destinationPoint.y()
                                                       )
            self.expectedConnection.setPen(QPen(Qt.Qt.DashLine))

            self.sceneContainerPt.addItem(self.expectedConnection)
        else:
            self.expectedConnection.setLine( sourcePoint.x()
                                           , sourcePoint.y()
                                           , destinationPoint.x()
                                           , destinationPoint.y()
                                           )
    def removeExpectedConnection(self):
        #print("removeExpectedConnection")
        self.sceneContainerPt.removeItem(self.expectedConnection)
        self.expectedConnection = None
        self.connectionSource   = None

    def removeConnector(self):
        if self.connectionSign is not None:
            self.sceneContainerPt.removeItem(self.connectionSign)
            self.connectionSign = None

    def showConnector(self, item):
        self.removeConnector()
        self.connectionSource = item
        rectangle = item.boundingRect()
        self.connectionSign = QGraphicsPixmapItem(
            QPixmap.fromImage(self.connectionSignImage.scaled( rectangle.height()
                                                             , rectangle.height() 
                                                             )))
        
        self.connectionSign.setParentItem(item.parentItem())
        self.connectionSign.setPos(0.0,0.0)
        position = item.mapToParent(rectangle.topRight())
        self.connectionSign.moveBy( position.x()
                                  , position.y() - rectangle.height() / 2.0
                                  )
        #print "self.connectionSign at showConnector",self.connectionSign
        # self.sceneContainerPt.addItem(self.connectionSign)
        self.connectionSign.setToolTip("Drag to connect.")

    def editorMouseReleaseEvent(self, event):
        if self.state["press"]["mode"] == INVALID:
            self.state["release"]["mode"] = INVALID
            self.resetState()
            return

        self.clickPosition  = self.mapToScene(event.pos())
        (item, itemType) = self.resolveItem(self.items(event.pos()), self.clickPosition)
        self.state["release"]["mode"] = VALID
        self.state["release"]["item"] = item
        self.state["release"]["type"] = itemType

        clickedItemType = self.state["press"]["type"]

        if clickedItemType == ITEM:
            #print("Adding Connector")
            self.showConnector(self.state["press"]["item"])
            # self.state["press"]["item"].setSelected(True)
            if not self.state["move"]["happened"]:
                self.layoutPt.plugin.mainWindow.objectEditSlot(self.state["press"]["item"].mobj, True)
            #print(self.state)

        if clickedItemType  == CONNECTOR :

            if self.state["move"]["happened"]:
                self.removeConnector()
                if isinstance(self.state["release"]["item"], KineticsDisplayItem):
                    #print("Connection created =>", self.connectionSource, self.state["release"]["item"])
                    self.populate_srcdes( self.connectionSource.mobj
                                        , self.state["release"]["item"].mobj
                                        )
                self.removeExpectedConnection()

            self.resetState()

        if clickedItemType == COMPARTMENT_BOUNDARY:
            if not self.state["move"]["happened"]:
                self.layoutPt.plugin.mainWindow.objectEditSlot(self.state["press"]["item"].mobj, True)
            self.resetState()

        if clickedItemType == COMPARTMENT_INTERIOR:
            if self.state["move"]["happened"]:
                startingPosition = self.state["press"]["pos"]
                endingPosition = event.pos()
                displacement   = endingPosition - startingPosition

                x0 = startingPosition.x() 
                x1 = endingPosition.x()
                y0 = startingPosition.y() 
                y1 = endingPosition.y()

                if displacement.x() < 0 :
                    x0,x1= x1,x0

                if displacement.y() < 0 :
                    y0,y1= y1,y0

                selectedItems = self.items(x0,y0,abs(displacement.x()), abs(displacement.y()))
                # print("Rect => ", self.customrubberBand.rect())
                # selectedItems = self.items(self.mapToScene(self.customrubberBand.rect()).boundingRect())
                self.selectSelections(selectedItems)
                #print("Rubberband Selections => ", self.selections)
                self.customrubberBand.hide()
                self.customrubberBand = None                
                popupmenu = QtGui.QMenu('PopupMenu', self)
                popupmenu.addAction("Delete", lambda : self.deleteSelections(x0,y0,x1,y1))
                popupmenu.addAction("Zoom", lambda : self.zoomSelections(x0, y0, x1, y1))
                popupmenu.addAction("Move", self.moveSelections)
                popupmenu.exec_(event.pos())
                # self.delete = QtGui.QAction(self.tr('delete'), self)
                # self.connect(self.delete, QtCore.SIGNAL('triggered()'), self.deleteItems)
                # self.zoom = QtGui.QAction(self.tr('zoom'), self)
                # self.connect(self.zoom, QtCore.SIGNAL('triggered()'), self.zoomItem)
                # self.move = QtGui.QAction(self.tr('move'), self)
                # self.connect(self.move, QtCore.SIGNAL('triggered()'), self.moveItem)
    



            # else:
            #     self.layoutPt.plugin.mainWindow.objectEditSlot(self.state["press"]["item"].mobj, True)

        self.resetState()



    def selectSelections(self, selections):
        for selection in selections :
            if isinstance(selection, KineticsDisplayItem):
                selection.setSelected(True)
                self.selections.append(selection)

    def deselectSelections(self):
        for selection in self.selections:
            selection.setSelected(False)
        self.selections  = []

    def mousePressEvent(self, event):
        selectedItem = None
        if self.viewBaseType == "editorView":
            return self.editorMousePressEvent(event)

        elif self.viewBaseType == "runView":
            pos = event.pos()
            item = self.itemAt(pos)
            if item:
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
        if self.viewBaseType == "editorView":
            return self.editorMouseMoveEvent(event)
        

    def mouseReleaseEvent(self, event):
        if self.viewBaseType == "editorView":
            return self.editorMouseReleaseEvent(event)

        return

        if self.state["press"]["mode"] == CONNECTION:
            desPos =self.mapToScene(event.pos())
            destination = self.items(event.pos())
            src = self.state["press"]["item"]
            des  = [j for j in destination if isinstance(j,KineticsDisplayItem)]
            if len(des):
                self.populate_srcdes(src.mobj,des[0].mobj)
                #print " pop", self.layoutPt.srcdesConnection()
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
            self.connect(self.delete, QtCore.SIGNAL('triggered()'), self.deleteItems)
            self.zoom = QtGui.QAction(self.tr('zoom'), self)
            self.connect(self.zoom, QtCore.SIGNAL('triggered()'), self.zoomItem)
            self.move = QtGui.QAction(self.tr('move'), self)
            self.connect(self.move, QtCore.SIGNAL('triggered()'), self.moveItem)
            popupmenu.addAction(self.delete)
            popupmenu.addAction(self.zoom)
            popupmenu.addAction(self.move)
            popupmenu.exec_(event.globalPos())
        self.showpopupmenu = False



    def moveSelections(self):
      self.setCursor(Qt.Qt.CrossCursor)
      self.deselectSelections()

    def GrVfitinView(self):
        #print " here in GrVfitinView"
        itemignoreZooming = False
        self.layoutPt.updateItemTransformationMode(itemignoreZooming)
        self.fitInView(self.sceneContainerPt.itemsBoundingRect().x()-10,self.sceneContainerPt.itemsBoundingRect().y()-10,self.sceneContainerPt.itemsBoundingRect().width()+20,self.sceneContainerPt.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
        self.layoutPt.drawLine_arrow(itemignoreZooming=False)
         

    def deleteSelections(self,x0,y0,x1,y1):
        #print "deleteSelections",x1-x0, " ",y1-y0
        vTransform = self.viewportTransform()
        self.deleted = False
        if( x1-x0 > 0  and y1-y0 >0):
            self.rubberbandlist = self.sceneContainerPt.items(self.mapToScene(QtCore.QRect(x0, y0, x1 - x0, y1 - y0)).boundingRect(), Qt.Qt.IntersectsItemShape)
            #print "self.rubberbandlist ",self.rubberbandlist
            for unselectitem in self.rubberbandlist:
                if unselectitem.isSelected() == True:
                    unselectitem.setSelected(0)
            for item in (qgraphicsitem for qgraphicsitem in self.rubberbandlist):
                #print "item for deletion ",item
                self.deleteItem(item)
                self.deleted = True
        # else:
        #     self.rubberbandlist = self.sceneContainerPt.items(self.endScenepos.x(),self.endScenepos.y(),abs(self.rubberbandWidth),abs(self.rubberbandHeight), Qt.Qt.IntersectsItemShape)
        #     for unselectitem in self.rubberbandlist:
        #         if unselectitem.isSelected() == True:
        #             unselectitem.setSelected(0)
        #     for items in (qgraphicsitem for qgraphicsitem in self.rubberbandlist):
        #         deleteItem(item)
        #         self.deleted = True
        if self.deleted:
            compt = moose.wildcardFind(self.modelRoot+'/##[ISA=ChemCompt]')
            comptinfo = moose.Annotator(moose.element(compt[0]).path+'/info')
            solver = comptinfo.solver
            if ( solver == 'gsl' ):
                ksolve = moose.Ksolve( compt[0].path+'/ksolve' )
            if ( solver == 'gssa' ):
                ksolve = moose.Gsolve( compt[0].path+'/ksolve' )
            if ( solver != 'ee' ):
                stoich = moose.Stoich( compt[0].path+'/stoich' )
                stoich.compartment = moose.element(compt[0])
                stoich.ksolve = ksolve
                stoich.path = compt[0].path+"/##"

        self.deselectSelections()

    def deleteItem(self,item):
        self.layoutPt.plugin.mainWindow.objectEditSlot('/',False)
        compt = moose.wildcardFind(self.modelRoot+'/##[ISA=ChemCompt]')
        
        if ( moose.exists( compt[0].path+'/stoich' ) ):
            moose.delete( compt[0].path+'/stoich' )
            moose.delete( compt[0].path+'/ksolve' )
        #print "item isinstance ",isinstance(item,KineticsDisplayItem)
        if isinstance(item,KineticsDisplayItem):
            if moose.exists(item.mobj.path):
                moose.delete(item.mobj)

                self.sceneContainerPt.removeItem(item)


        elif isinstance(item,QtGui.QGraphicsPolygonItem):
            self.sceneContainerPt.removeItem(item)

    def zoomSelections(self, x0, y0, x1, y1):

        # for selection in self.selections:
        #     if isinstance(selection, PoolItem):

        self.fitInView(self.mapToScene(QtCore.QRect(x0, y0, x1 - x0, y1 - y0)).boundingRect(), Qt.Qt.KeepAspectRatio)
        #print " matrix",self.matrix().m11()
        self.deselectSelections()

        # vTransform = self.viewportTransform()
        # rubberbandWidth     = x1 - x0
        # rubberbandHeight    = y1 - y0

        # for selection in self.selections:
        #     if isinstance(selection, PoolItem):
        #     if((self.matrix().m11()>=1.0)and(self.matrix().m22() >=1.0)):
        #         for item in ( Txtitem for Txtitem in self.sceneContainerPt.items() if isinstance(Txtitem,PoolItem) ):
        #             item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, False)
        # else:
        #     self.rubberbandlist = self.sceneContainerPt.items(self.endScenepos.x(),self.endScenepos.y(),abs(self.rubberbandWidth),abs(self.rubberbandHeight), Qt.Qt.IntersectsItemShape)
        #     for unselectitem in self.rubberbandlist:
        #         if unselectitem.isSelected() == True:
        #             unselectitem.setSelected(0)
        #     for items in (qgraphicsitem for qgraphicsitem in self.rubberbandlist if isinstance(qgraphicsitem,PoolItem)):
        #         self.fitInView(self.endScenepos.x(),self.endScenepos.y(),abs(self.rubberbandWidth),abs(self.rubberbandHeight),Qt.Qt.KeepAspectRatio)
        #         if((self.matrix().m11()>=1.0)and(self.matrix().m22() >=1.0)):
        #             for item in ( Txtitem for Txtitem in self.sceneContainerPt.items() if isinstance (Txtitem, PoolItem)):
        #                 item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, False)
        # self.rubberBandactive = False


    # def resizeEvent(self, event):
    # #     """ zoom when resize! """
    #     self.fitInView(self.sceneContainerPt.itemsBoundingRect().x()-10,self.sceneContainerPt.itemsBoundingRect().y()-10,self.sceneContainerPt.itemsBoundingRect().width()+20,self.sceneContainerPt.itemsBoundingRect().height()+20,Qt.Qt.IgnoreAspectRatio)
    #     # QtGui.QGraphicsView.resizeEvent(self, event)

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
            #print "event.scenePostion ",event.scenePos()
            string = str(event.mimeData().text())
            createObj(self.viewBaseType,self,self.modelRoot,string,event_pos,self.layoutPt)

    def populate_srcdes(self,src,des):
        self.modelRoot = self.layoutPt.modelRoot
        callsetupItem = True
        #print " populate_srcdes ",src,des
        if ( isinstance(moose.element(src),PoolBase) and ( (isinstance(moose.element(des),ReacBase) ) or isinstance(moose.element(des),EnzBase) )):
            moose.connect(src, 'reac', des, 'sub', 'OneToOne')
        elif(isinstance (moose.element(src),PoolBase) and (isinstance(moose.element(des),Function))):
            moose.connect( src, 'nOut', des.x[des.numVars-1], 'input' )
            if (des.numVars-1) == 0:
                des.expr ='x'+str(des.numVars-1)
            else:
                des.expr = des.expr+'+x'+str(des.numVars-1)
            des.numVars+=1
        elif( isinstance(moose.element(src),Function) and (moose.element(des).className=="Pool") ):
                if ((element(des).parent).className != 'Enz'):
                    moose.connect(src, 'valueOut', des, 'increment', 'OneToOne')
                else:
                    srcdesString = element(src).className+'-- EnzCplx'
                    QtGui.QMessageBox.information(None,'Connection Not possible','\'{srcdesString}\' not allowed to connect'.format(srcdesString = srcdesString),QtGui.QMessageBox.Ok)
                    callsetupItem = False
        elif( isinstance(moose.element(src),Function) and (moose.element(des).className=="BufPool") ):
                moose.connect(src, 'valueOut', des, 'setConcInit', 'OneToOne')
        elif( isinstance(moose.element(src),Function) and (isinstance(moose.element(des),ReacBase) ) ):
                moose.connect(src, 'valueOut', des, 'setNumKf', 'OneToOne')
        elif( isinstance(moose.element(src),ReacBase) and (isinstance(moose.element(des),PoolBase) ) ):
            moose.connect(src, 'prd', des, 'reac', 'OneToOne')
        elif( isinstance(moose.element(src),EnzBase) and (isinstance(moose.element(des),PoolBase) ) ):
            moose.connect(src, 'prd', des, 'reac', 'OneToOne')
        elif( isinstance(moose.element(src),StimulusTable) and (isinstance(moose.element(des),PoolBase) ) ):
            moose.connect(src, 'output', des, 'setConcInit', 'OneToOne')
        else:
            srcString = moose.element(src).className
            desString = moose.element(des).className
            srcdesString = srcString+'--'+desString
            # QtGui.QMessageBox.information(None,'Connection Not possible','\'{srcString}\' Not allowed to connect \'{desString}\' '.format(srcString = srcString, desString=desString),QtGui.QMessageBox.Ok)
            QtGui.QMessageBox.information(None,'Connection Not possible','\'{srcdesString}\' not allowed to connect'.format(srcdesString = srcdesString),QtGui.QMessageBox.Ok)
            callsetupItem = False
            
        if callsetupItem:
            #print " here in kkitView ",self.layoutPt.srcdesConnection
            setupItem(self.modelRoot,self.layoutPt.srcdesConnection)
            #print " after ",self.layoutPt.srcdesConnection
            self.layoutPt.drawLine_arrow(False)

