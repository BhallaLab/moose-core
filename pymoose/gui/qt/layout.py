# LayoutWidget for Loading Genesis files

import sys
import os
from PyQt4 import QtGui,QtCore,Qt
from operator import itemgetter, attrgetter
import math
import config
import moose
'''
c = moose.PyMooseBase.getContext()
c.loadG('/home/lab13/Genesis_file/gfile/acc25.g')
app = QtGui.QApplication(sys.argv)
'''
class Textitem(QtGui.QGraphicsTextItem): 
	positionChange = QtCore.pyqtSignal(QtGui.QGraphicsItem)
	def __init__(self,parent,path):
		self.mooseObj_ = moose.Neutral(path)
		if isinstance (parent, LayoutWidget):
			QtGui.QGraphicsTextItem.__init__(self,self.mooseObj_.name)
		elif isinstance (parent,Rect_Compt):
			QtGui.QGraphicsTextItem.__init__(self,parent)
			
		self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
		self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
		self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, 1) 
		if config.QT_MINOR_VERSION >= 6:
		 	self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, 1) 
	
	def itemChange(self, change, value):
		if change == QtGui.QGraphicsItem.ItemPositionChange:
			self.positionChange.emit(self)
       		return QtGui.QGraphicsItem.itemChange(self, change, value)
	
	def mouseDoubleClickEvent(self, event):
		self.emit(QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.mooseObj_)
	
	def updateSlot(self):
		if(self.mooseObj_.className == 'Enzyme'):
			textColor = "<html><body bgcolor='black'>"+self.mooseObj_.name+"</body></html>"
			self.setHtml(textColor)
		else:
			textColor = self.mooseObj_.getField('xtree_fg_req')
			self.layoutWidgetpt.colorCheck(self,self.mooseObj_,textColor,"background")

class Rect_Compt(QtGui.QGraphicsRectItem):
	def __init__(self,layoutwidget,x,y,w,h,path):
		self.mooseObj_ = moose.Neutral(path)
		self.layoutWidgetpt = layoutwidget
		self.Rectemitter = QtCore.QObject()
		QtGui.QGraphicsRectItem.__init__(self,x,y,w,h)
		self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
		self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable)
		
		self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, 1) 
		if config.QT_MINOR_VERSION >= 6:
		 	self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, 1) 
		 	
	def mouseDoubleClickEvent(self, event):
		self.Rectemitter.emit(QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.mooseObj_)
	
	def itemChange(self, change, value):
		if change == QtGui.QGraphicsItem.ItemPositionChange:
			self.Rectemitter.emit(QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.mooseObj_)
       		return QtGui.QGraphicsItem.itemChange(self, change, value)
	
	def updateSlot(self):
		if(self.mooseObj_.className == 'Enzyme'):
			textColor = "<html><body bgcolor='black'>"+self.mooseObj_.name+"</body></html>"
			self.setHtml(textColor)
		else:
			textColor = self.mooseObj_.getField('xtree_fg_req')
			self.layoutWidgetpt.colorCheck(self,self.mooseObj_,textColor,"background")
			
class Graphicalview(QtGui.QGraphicsView):
	def __init__(self,scenecontainer,border):
		self.sceneContainerPt = scenecontainer
		self.border = border
		QtGui.QGraphicsView.__init__(self,self.sceneContainerPt)
		self.setDragMode(QtGui.QGraphicsView.RubberBandDrag)
		self.setScene(self.sceneContainerPt)
		self.rubberBandactive = False
		self.itemSelected = False
		self.customrubberBand=0
		self.rubberbandWidth = 0
		self.rubberbandHeight = 0
			
	def mousePressEvent(self, event):
		
		if event.buttons() == QtCore.Qt.RightButton and self.rubberBandactive == True:
			popupmenu = QtGui.QMenu('PopupMenu', self)
			self.delete = QtGui.QAction(self.tr('delete'), self)
			self.connect(self.delete, QtCore.SIGNAL('triggered()'), self.deleteItem)
			
			self.zoom = QtGui.QAction(self.tr('zoom'), self)
			self.connect(self.zoom, QtCore.SIGNAL('triggered()'), self.zoomItem)
			
			popupmenu.addAction(self.delete)
			popupmenu.addAction(self.zoom)
			popupmenu.exec_(event.globalPos())

		if event.buttons() == QtCore.Qt.LeftButton:
			self.startingPos = event.pos()
			self.startScenepos = self.mapToScene(self.startingPos)
			
			self.deviceTransform = self.viewportTransform()
			if config.QT_MINOR_VERSION >= 6:
				''' deviceTransform needs to be provided if the scene contains items that ignore transformations,
					which was introduced in 4.6
				'''
				sceneitems = self.sceneContainerPt.itemAt(self.startScenepos,self.deviceTransform)
			else:
				''' for below  Qt4.6 there is no view transform for itemAt 
				     and if view is zoom out below 50%  and if textitem object is moved, 
				     along moving the item, zooming also happens.
				'''
				sceneitems = self.sceneContainerPt.itemAt(self.startScenepos)
					
			
			
			#checking if rubberband selection start on any item (in my case textitem or rectcompartment) if none, start the rubber effect 
			if ( sceneitems == None):
				QtGui.QGraphicsView.mousePressEvent(self, event)
				self.itemSelected = False
				
			#Since qgraphicsrectitem is a item in qt, if I select inside the rectangle it would select the entire rectangle
			# and would not allow me to select the items inside the rectangle so breaking the code by not calling parent class to inherit functionality
			#rather writing custom code for rubberband effect here
			elif( sceneitems != None):
				
				if(isinstance(sceneitems, Textitem)):
					QtGui.QGraphicsView.mousePressEvent(self, event)
					self.itemSelected = True
				elif(isinstance(sceneitems, Rect_Compt)):
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
					if(((xp > xs-self.border/2) and (xp < xs+self.border/2) and (yp > ys-self.border/2) and (yp < ye+self.border/2) )or 
					   ((xp > xs+self.border/2) and (xp < xe-self.border/2) and (yp > ye-self.border/2) and (yp < ye+self.border/2) ) or 
					   ((xp > xs+self.border/2) and (xp < xe-self.border/2) and (yp > ys-self.border/2) and (yp < ys+self.border/2) ) or
					   ((xp > xe-self.border/2) and (xp < xe+self.border/2) and (yp > ys-self.border/2) and (yp < ye+self.border/2) ) ):
					   	if sceneitems.isSelected() == False:
					   		sceneitems.setSelected(1)
					   		self.itemSelected = True
						QtGui.QGraphicsView.mousePressEvent(self, event)

					else:
						#if its inside the qgraphicsrectitem then custom code for starting rubberband selection	
						self.rubberBandactive = True
						self.itemSelected = False
						self.customrubberBand = QtGui.QRubberBand(QtGui.QRubberBand.Rectangle,self)
						self.customrubberBand.setGeometry(QtCore.QRect(self.startingPos,QtCore.QSize()))
						self.customrubberBand.show()
				else:
					print "Report this functionality may not be implimentated"
					
	def mouseMoveEvent(self,event):
		QtGui.QGraphicsView.mouseMoveEvent(self, event)

		if self.customrubberBand == 0 and event.buttons() == QtCore.Qt.LeftButton and self.itemSelected == False :
			self.rubberBandactive = True

		elif event.buttons() == QtCore.Qt.LeftButton and self.itemSelected == True :
			for selecteditem in self.sceneContainerPt.selectedItems():
				self.emit(QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),selecteditem.mooseObj_)
				self.rubberBandactive = False
							
		if( (self.customrubberBand) and (event.buttons() == QtCore.Qt.LeftButton)):
				self.endingPos = event.pos()
				self.endingScenepos = self.mapToScene(self.endingPos)
				self.rubberbandWidth = self.endingScenepos.x()-self.startScenepos.x()
				self.rubberbandHeight = self.endingScenepos.y()-self.startScenepos.y()

				
				self.customrubberBand.setGeometry(QtCore.QRect(self.startingPos, event.pos()).normalized())
								
				#unselecting any previosly selected item in scene
				for preSelectItem in self.sceneContainerPt.selectedItems():
					preSelectItem.setSelected(0)
					
				for items in self.sceneContainerPt.items(self.startScenepos.x(),self.startScenepos.y(),self.rubberbandWidth,self.rubberbandHeight,Qt.Qt.IntersectsItemShape):
					if(isinstance(items,Textitem)):
						if items.isSelected() == False:
							items.setSelected(1)
						
	def mouseReleaseEvent(self, event):
						
		QtGui.QGraphicsView.mouseReleaseEvent(self, event)
		if(self.customrubberBand):
			self.customrubberBand.hide()
			self.customrubberBand = 0
		
		if((event.button() == QtCore.Qt.LeftButton) and (self.rubberBandactive == True)):
			self.endingPos = event.pos()
			self.endScenepos = self.mapToScene(self.endingPos)
			self.rubberbandWidth = (self.endScenepos.x()-self.startScenepos.x())
			self.rubberbandHeight = (self.endScenepos.y()-self.startScenepos.y())
	
	def deleteItem(self):
		self.rubberBandactive = False
		for items in self.sceneContainerPt.selectedItems():
			print "items to be deleted",items.mooseObj_.name
	
	def zoomItem(self):
		
		vTransform = self.viewportTransform()
		if( self.rubberbandWidth > 0  and self.rubberbandHeight >0):
			self.rubberbandlist = self.sceneContainerPt.items(self.startScenepos.x(),self.startScenepos.y(),self.rubberbandWidth,self.rubberbandHeight, Qt.Qt.IntersectsItemShape)
			for unselectitem in self.rubberbandlist:
				if unselectitem.isSelected() == True:
					unselectitem.setSelected(0)
	
			for items in (qgraphicsitem for qgraphicsitem in self.rubberbandlist if isinstance(qgraphicsitem,Textitem)):
				self.fitInView(self.startScenepos.x(),self.startScenepos.y(),self.rubberbandWidth,self.rubberbandHeight,Qt.Qt.KeepAspectRatio)
				if((self.matrix().m11()>=1.0)and(self.matrix().m22() >=1.0)):
					for item in ( Txtitem for Txtitem in self.sceneContainerPt.items() if isinstance (Txtitem, Textitem)):
						item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, False)
		else:
			self.rubberbandlist = self.sceneContainerPt.items(self.endScenepos.x(),self.endScenepos.y(),abs(self.rubberbandWidth),abs(self.rubberbandHeight), Qt.Qt.IntersectsItemShape)
			for unselectitem in self.rubberbandlist:
				if unselectitem.isSelected() == True:
					unselectitem.setSelected(0)
	
			for items in (qgraphicsitem for qgraphicsitem in self.rubberbandlist if isinstance(qgraphicsitem,Textitem)):
				self.fitInView(self.endScenepos.x(),self.endScenepos.y(),abs(self.rubberbandWidth),abs(self.rubberbandHeight),Qt.Qt.KeepAspectRatio)
				if((self.matrix().m11()>=1.0)and(self.matrix().m22() >=1.0)):
					for item in ( Txtitem for Txtitem in self.sceneContainerPt.items() if isinstance (Txtitem, Textitem)):
						item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, False)		
		self.rubberBandactive = False
	
class Widgetvisibility(Exception):pass

class LayoutWidget(QtGui.QWidget):
	def __init__(self,parent=None):
		QtGui.QWidget.__init__(self,parent)
		grid = QtGui.QGridLayout()
		self.setLayout(grid)
		self.setWindowTitle('Layout')		
		self.sceneContainer = QtGui.QGraphicsScene(self)
		self.sceneContainer.setBackgroundBrush(QtGui.QColor(230,230,219,120))
		#~ self.sceneContainer.setSceneRect(self.sceneContainer.itemsBoundingRect())
	
		self.rootObject = moose.Neutral('/kinetics')
		self.itemList = []
		self.setupItem(self.rootObject,self.itemList)
		self.moosetext_dict = {}
		self.Compt_dict = {}
		self.border = 10
		#colorMap of kinetic kit
		self.colorMap = ((248,0,255),(240,0,255),(232,0,255),(224,0,255),(216,0,255),(208,0,255),(200,0,255),(192,0,255),(184,0,255),(176,0,255),(168,0,255),(160,0,255),(152,0,255),(144,0,255),(136,0,255),(128,0,255),(120,0,255),(112,0,255),(104,0,255),(96,0,255),(88,0,255),(80,0,255),(72,0,255),(64,0,255),(56,0,255),(48,0,255),(40,0,255),(32,0,255),(24,0,255),(16,0,255),(8,0,255),(0,0,255),(0,8,248),(0,16,240),(0,24,232),(0,32,224),(0,40,216),(0,48,208),(0,56,200),(0,64,192),(0,72,184),(0,80,176),(0,88,168),(0,96,160),(0,104,152),(0,112,144),(0,120,136),(0,128,128),(0,136,120),(0,144,112),(0,152,104),(0,160,96),(0,168,88),(0,176,80),(0,184,72),(0,192,64),(0,200,56),(0,208,48),(0,216,40),(0,224,32),(0,232,24),(0,240,16),(0,248,8),(0,255,0),(8,255,0),(16,255,0),(24,255,0),(32,255,0),(40,255,0),(48,255,0),(56,255,0),(64,255,0),(72,255,0),(80,255,0),(88,255,0),(96,255,0),(104,255,0),(112,255,0),(120,255,0),(128,255,0),(136,255,0),(144,255,0),(152,255,0),(160,255,0),(168,255,0),(176,255,0),(184,255,0),(192,255,0),(200,255,0),(208,255,0),(216,255,0),(224,255,0),(232,255,0),(240,255,0),(248,255,0),(255,255,0),(255,248,0),(255,240,0),(255,232,0),(255,224,0),(255,216,0),(255,208,0),(255,200,0),(255,192,0),(255,184,0),(255,176,0),(255,168,0),(255,160,0),(255,152,0),(255,144,0),(255,136,0),(255,128,0),(255,120,0),(255,112,0),(255,104,0),(255,96,0),(255,88,0),(255,80,0),(255,72,0),(255,64,0),(255,56,0),(255,48,0),(255,40,0),(255,32,0),(255,24,0),(255,16,0),(255,8,0),(255,0,0))
		
		#This is check which version of kkit, b'cos anything below kkit8 didn't had xyz co-ordinates
		allZero = "True"
		for item in self.itemList:
			x = float( item.getField( 'x' ) )
			y = float( item.getField( 'y' ) )
			
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
			for item in self.itemList:
			
				#Creating all the compartments for the model and adding to the scene
				self.key = moose.Neutral(item.parent)
				self.create_Compt(item,self.key)
				
				#Adding moose Object to scene to get scene bounding reaction for spacing the coordinates	
				pItem = Textitem(self,item.path)
				x = float(item.getField('x'))
				y = float(item.getField('y'))
				pItem.setPos(x,y)
				itemid = item.id
				
				#moosetext_dict[] is created to get the text's sceneBoundingRect
				self.moosetext_dict[itemid] = pItem 
					
		#Calculate the scaling factor for cordinates		
		self.scale_Cord = int(self.cordTransform(self.itemList,self.moosetext_dict))		
		
		#Adding moose Object to scene and then adding to scene to view
		for item in self.itemList:
				textColor = ""
				textBgcolor=""
				textColor = item.getField('xtree_textfg_req')
				textBgcolor = item.getField('xtree_fg_req')
				x = float(item.getField('x'))*(self.scale_Cord)
				y = float(item.getField('y'))*-(self.scale_Cord)
				
				self.key = moose.Neutral(item.parent)
				if self.key.className == 'KinCompt':
					value = self.Compt_dict[self.key.name]
					pItem = Textitem(value,item.path)
					pItem.setPos(x,y)
				elif self.key.className == 'KineticManager':	
					pItem = Textitem(self,item.path)
					pItem.setPos(x,y)
					self.sceneContainer.addItem(pItem)
				else:
					tobfoundkey = self.key.parent
					found = 1
					number=0
					while found == 1 and number < 3:
						self.key = moose.Neutral(tobfoundkey)
						number = number+1
						if self.key.className == 'KinCompt':
							value = self.Compt_dict[self.key.name]
							pItem = Textitem(value,item.path)
							pItem.setPos(x,y)	
							found = 0
						elif self.key.className == 'KineticManager':
							pItem = Textitem(self,item.path)
							pItem.setPos(x,y)	
							self.sceneContainer.addItem(pItem)
							found = 0	
						else:
							tobfoundkey= self.key.parent
				self.connect(pItem, QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"), self.emitItemtoEditor)
				pItem.positionChange.connect(self.positionChange)
								
				itemid = item.id
				self.moosetext_dict[itemid] = pItem 	
						
				if(item.className =='Enzyme'):	
					parent = moose.Neutral(item.parent)
					textParcolor = moose.Neutral(parent.path).getField('xtree_fg_req')
					self.colorCheck(pItem,item,textParcolor,"foreground")
					textbgcolor = "<html><body bgcolor='black'>"+item.name+"</body></html>"
					pItem.setHtml(QtCore.QString(textbgcolor))	
				
				else:
					if(textColor==textBgcolor):
						textBgcolor="black"
					self.colorCheck(pItem,item,textColor,"foreground")
					self.colorCheck(pItem,item,textBgcolor,"background")
			
		 #RectCompartment which is kincompartment is added to the screne
		for k, v in self.Compt_dict.items():
				rectcompt = v.childrenBoundingRect()
	
				v.setRect(rectcompt.x()-10,rectcompt.y()-10,(rectcompt.width()+20),(rectcompt.height()+20))
				v.setPen( QtGui.QPen( Qt.QColor(66,66,66,100), self.border, QtCore.Qt.SolidLine,QtCore.Qt.RoundCap,QtCore.Qt.RoundJoin ) )
				v.Rectemitter.connect(v.Rectemitter,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.positionChange)
				v.Rectemitter.connect(v.Rectemitter,QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.emitItemtoEditor)

		#connecting substrate,product to reaction and Enzyme
		self.lineItem_dict = {}
		self.object2line = {}
		
		for item in self.itemList:
			src = ""
			des = ""
			if(item.className != "Molecule"):
				for sobject in moose.Neutral(item.path).neighbours('sub', 0):
					src = self.moosetext_dict[item.id]
					des = self.moosetext_dict[sobject]
					self.lineCord(src,des,item)
				for pobject in moose.Neutral(item.path).neighbours('prd', 0):
					src = self.moosetext_dict[pobject]
					des = self.moosetext_dict[item.id]
					self.lineCord(src,des,item)	
				
			#Added to substrate and product for reaction, we need to take enz also for the enzymetic Reaction	
			if(item.className == "Enzyme"):
				for pobject in moose.Neutral(item.path).neighbours('enz',0): 
					src = self.moosetext_dict[pobject]
					des = self.moosetext_dict[item.id]
					self.lineCord(src,des,item)
		
		sceneBoundRect = self.sceneContainer.itemsBoundingRect()
		self.view = Graphicalview(self.sceneContainer,self.border)
		self.view.setScene(self.sceneContainer)
		self.view.centerOn(self.sceneContainer.sceneRect().center())
		grid.addWidget(self.view,0,0)
	
	# setting up moose item	
	def setupItem(self,mooseObject,itemlist):
		for child in mooseObject.children():
			childObj = moose.Neutral(child)
			if( (childObj.className == 'Molecule') | (childObj.className == 'Reaction') |(childObj.className == 'Enzyme') ):
				''' This is for eliminating enzyme complex'''
				if((moose.Neutral(childObj.parent).className) != 'Enzyme'):
					itemlist.append(childObj)
			self.setupItem(childObj,itemlist)
	
	#Checking and creating compartment
	def create_Compt(self,item,key):
		rectitem = ""
		 
		if (key.className == 'KinCompt'):
			if self.Compt_dict.has_key(key.name):
				rectitem = self.Compt_dict[key.name]
			else:
				self.new_Compt = Rect_Compt(self,0,0,0,0,key.path)
				self.Compt_dict[key.name] = self.new_Compt
				self.new_Compt.setRect(10,10,10,10)
				self.sceneContainer.addItem(self.new_Compt)
		elif (key.className == 'KineticManager'):
			rectitem = 'kinetics'
		else:
			parent_key = moose.Neutral(key.parent)
			rectitem = self.create_Compt(item,parent_key)
			
	def cordTransform(self,itemslist,mooseItemdict):
		#here alpha is calculated to multipy the coordinates with, so that the each items spreads out
		alpha = 0
		alpha1 = 0
		for t in range(len(itemslist)):
			src = itemslist[t]
			self.pitemsrc = mooseItemdict[src.id]
			src_Scenebounding = self.pitemsrc.sceneBoundingRect()
			srcX = src_Scenebounding.x()
			srcY = src_Scenebounding.y()
			#print "item bounding",srcX,srcY,src.name,srcX+srcW,srcY+srcH
		for m in range(len(itemslist)):
			for n in range(len(itemslist)):
				if(m != n):
					src = itemslist[m]
					self.pitemsrc = mooseItemdict[src.id]
					src_Scenebounding = self.pitemsrc.sceneBoundingRect()
					srcX = src_Scenebounding.x()
					srcY = src_Scenebounding.y()
					srcW = src_Scenebounding.right()-src_Scenebounding.left()
					srcH = src_Scenebounding.bottom()-src_Scenebounding.top()

					des = itemslist[n]
					self.pitemdes = mooseItemdict[des.id]
					des_Scenebounding = self.pitemdes.sceneBoundingRect()
					desX = des_Scenebounding.x()
					desY = des_Scenebounding.y()
					desW = des_Scenebounding.right()-des_Scenebounding.left()
					desH = des_Scenebounding.bottom()-des_Scenebounding.top()
					
					t = src_Scenebounding.intersects(des_Scenebounding)
					if t:
						sfx = 0
						sfy = 0
						if((desX - srcX)!= 0):	sfx = ( float(srcW)/abs(desX-srcX))
						if((desY - srcY)!= 0):	sfy = ( float(srcH)/abs(desY-srcY))
						if((sfx != 0) and (sfy != 0)):
							if( sfx < sfy):		alpha = sfx
							elif (sfy < sfx):	alpha = sfy
							else:			alpha = 0
						elif (sfx == 0): alpha = sfy
						elif (sfy == 0): alpha = sfx
						else:		 alpha =0
					else:
						pass
					
					if(alpha1 < alpha): alpha1 = alpha
					
				else: 
					pass
		alpha1=alpha1+1
		return(alpha1)
	#color map for kinetic kit
	def colorCheck(self,pItem,item,textColor,fgbg_color):
		if(textColor == "<blank-string>"): textColor = "green"		
		if textColor.isdigit():
			tc = int(textColor)
			tc = (tc * 2 )
			r,g,b = self.colorMap[tc]
			if(fgbg_color == 'foreground'):
				pItem.setDefaultTextColor(QtGui.QColor(r,g,b))
			elif(fgbg_color == 'background'):
				hexchars = "0123456789ABCDEF"
				hexno = "#" + hexchars[r / 16] + hexchars[r % 16] + hexchars[g / 16] + hexchars[g % 16] + hexchars[b / 16] + hexchars[b % 16]	
				textbgcolor = "<html><body bgcolor="+hexno+">"+item.name+"</body></html>"
				pItem.setHtml(QtCore.QString(textbgcolor))
		else:	
			if(fgbg_color == 'foreground'):
				pItem.setDefaultTextColor(QtGui.QColor(textColor))
			elif(fgbg_color == 'background'):
				textbgcolor = "<html><body bgcolor='"+textColor+"'>"+item.name+"</body></html>"
				pItem.setHtml(QtCore.QString(textbgcolor))
	# Calculating line distance
	def lineCord(self,src,des,source):
		if( (src == "") & (des == "") ):
			print "Source or destination is missing or incorrect"
		
		else:
			srcdes_list= [src,des]
			arrow = self.calArrow(src,des)
		
			if(source.className == "Reaction"):
				qgLineitem = self.sceneContainer.addPolygon(arrow,QtGui.QPen(QtCore.Qt.green, 1, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))

			elif(source.className == "Enzyme"):
				qgLineitem = self.sceneContainer.addPolygon(arrow,QtGui.QPen(QtCore.Qt.red, 1, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
			
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
									
	def calArrow(self,src,des):
		sX = src.sceneBoundingRect().x()
		sY = src.sceneBoundingRect().y()			
		sw = src.sceneBoundingRect().right() -src.sceneBoundingRect().left()
		sh = src.sceneBoundingRect().bottom() -src.sceneBoundingRect().top()
		
		dX = des.sceneBoundingRect().x()
		dY = des.sceneBoundingRect().y()			
		dw = des.sceneBoundingRect().right() -des.sceneBoundingRect().left()
		dh = des.sceneBoundingRect().bottom() -des.sceneBoundingRect().top()

		#Here there is external boundary created for each textitem 
		#	1. for checking if there is overLap
		#	2. The start line and arrow head ends to this outer boundary
		
		srcRect = QtCore.QRectF(sX-5,sY-5,sw+10,sh+10)
		desRect = QtCore.QRectF(dX-5,dY-5,dw+10,dh+10)
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
		
		lItemSP = QtCore.QLineF(X-5,Y-5,X+w+5,Y-5)
		boundintersect= lItemSP.intersect(centerLine,linePoint)
		
		if (boundintersect == 1):
			return centerLine.angle()
			
		else:
			lItemSP = QtCore.QLineF(X+w+5,Y-5,X+w+5,Y+h+5)
			boundintersect= lItemSP.intersect(centerLine,linePoint)
			if (boundintersect == 1):
				return centerLine.angle()
			else:
				lItemSP = QtCore.QLineF(X+w+5,Y+h+5,X-5,Y+h+5)
				boundintersect= lItemSP.intersect(centerLine,linePoint)
				if (boundintersect == 1):
					return centerLine.angle()
					
				else:
					lItemSP = QtCore.QLineF(X-5,Y+h+5,X-5,Y-5)
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
	
	def positionChange(self,mooseObject):
		#If the item position changes, the corresponding arrow's are claculated
		
		if(isinstance(mooseObject, Textitem)):
				self.updatearrow(mooseObject)
		else:
			for k, v in self.Compt_dict.items():
					for rectChilditem in v.childItems():
						self.updatearrow(rectChilditem)
							
	def emitItemtoEditor(self,mooseObject):
		self.emit(QtCore.SIGNAL("itemDoubleClicked(PyQt_PyObject)"), mooseObject)
	
	def updatearrow(self,mooseObject):
		listItem = []
		for listItem in (v for k,v in self.object2line.items() if k.mooseObj_.id == mooseObject.mooseObj_.id ):
			if len(listItem):
				for ql,va in listItem:
					srcdes = self.lineItem_dict[ql]
					arrow = self.calArrow(srcdes[0],srcdes[1])
					ql.setPolygon(arrow)
			break
	
	def updateItemSlot(self, mooseObject):
			#In this case if the name is updated from the keyboard both in mooseobj and gui gets updation
	        for changedItem in (item for item in self.sceneContainer.items() if isinstance(item, Textitem) and mooseObject.id == item.mooseObj_.id):
	            break
	        changedItem.updateSlot()
	        self.positionChange(changedItem.mooseObj_)
	        
	def keyPressEvent(self,event):
		#For Zooming
		for item in self.sceneContainer.items():
			
			
			if isinstance (item, Textitem):
				if((self.view.matrix().m11()<=1.0)and(self.view.matrix().m22() <=1.0)):
					item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, True)
				else:
					item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, False)
			elif isinstance(item,Rect_Compt):
					pass
					#~ print "rect item",item.childrenBoundingRect()	
		key = event.key()
		
		if key == QtCore.Qt.Key_A:
			self.view.resetMatrix()
	
		elif (key == 46 or key == 62):
			print "here in sceneContainer",self.sceneContainer.sceneRect()
			self.view.scale(1.1,1.1)
			
		elif (key == 44 or key == 60):	
			print "here in sceneContainer",self.sceneContainer.sceneRect()
			self.view.scale(1/1.1,1/1.1)				
			
if __name__ == "__main__":
	#app = QtGui.QApplication(sys.argv)
	dt = LayoutWidget()
	dt.show()
	sys.exit(app.exec_())
		
