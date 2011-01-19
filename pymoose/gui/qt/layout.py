import sys
import os
from PyQt4 import QtGui,QtCore,Qt
from operator import itemgetter, attrgetter

import math
import moose

'''
#import config

print os.getcwd()
os.chdir('/home/lab13/trunk/pymoose/gui')
sys.path.append('/home/lab13/trunk/pymoose/')

import moose
dir(moose)

c = moose.PyMooseBase.getContext()

c.loadG('/home/lab13/Genesis_file/gfile/acc13.g')
#c.loadG('/home/lab13/trunk/DEMOS/kholodenko/Kholodenko.g')

app = QtGui.QApplication(sys.argv)
'''

class Textitem(QtGui.QGraphicsTextItem):
	def __init__(self,layoutwidget,path):
		self.mooseObj_ = moose.Neutral(path)
		self.layoutWidgetpt = layoutwidget
		QtGui.QGraphicsTextItem.__init__(self,self.mooseObj_.name)
		self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
		
	def mouseDoubleClickEvent(self, event):
		self.emit(QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.mooseObj_)
	
	def mousePressEvent(self, event):
		QtGui.QGraphicsTextItem.mousePressEvent(self, event)
		if event.button() == QtCore.Qt.LeftButton:
			self.startpos = event.scenePos()
		
	def mouseMoveEvent(self,event):
		QtGui.QGraphicsTextItem.mouseMoveEvent(self, event)
		if event.buttons() == QtCore.Qt.LeftButton:
			self.endpos = event.scenePos()
			if ((self.endpos.x()-self.startpos.x() != 0) |(self.endpos.y()-self.startpos.y() != 0)):
				self.emit(QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.mooseObj_)

	def mouseReleaseEvent(self, event):
		QtGui.QGraphicsTextItem.mouseReleaseEvent(self, event)

	def updateSlot(self):
		if(self.mooseObj_.className == 'Enzyme'):
			textColor = "<html><body bgcolor='black'>"+self.mooseObj_.name+"</body></html>"
			self.setHtml(textColor)
		else:
			textColor = self.mooseObj_.getField('xtree_fg_req')
			self.layoutWidgetpt.colorCheck(self,self.mooseObj_,textColor,"background")

class Widgetvisibility(Exception):pass

class LayoutWidget(QtGui.QWidget):
	def __init__(self,parent=None):
		QtGui.QWidget.__init__(self,parent)
		grid = QtGui.QGridLayout()
		self.setLayout(grid)
		self.screen = QtGui.QGraphicsScene(self)
		self.screen.setBackgroundBrush(QtGui.QColor(230,230,219,120))
		self.screen.setSceneRect(self.screen.itemsBoundingRect())
	
		self.rootObject = moose.Neutral('/kinetics')
		self.itemList = []
		self.setupItem(self.rootObject,self.itemList)
		self.moosetext_dict = {}
		
		#colorMap of kinetic kit
		self.colorMap = ((248,0,255),(240,0,255),(232,0,255),(224,0,255),(216,0,255),(208,0,255),(200,0,255),(192,0,255),(184,0,255),(176,0,255),(168,0,255),(160,0,255),(152,0,255),(144,0,255),(136,0,255),(128,0,255),(120,0,255),(112,0,255),(104,0,255),(96,0,255),(88,0,255),(80,0,255),(72,0,255),(64,0,255),(56,0,255),(48,0,255),(40,0,255),(32,0,255),(24,0,255),(16,0,255),(8,0,255),(0,0,255),(0,8,248),(0,16,240),(0,24,232),(0,32,224),(0,40,216),(0,48,208),(0,56,200),(0,64,192),(0,72,184),(0,80,176),(0,88,168),(0,96,160),(0,104,152),(0,112,144),(0,120,136),(0,128,128),(0,136,120),(0,144,112),(0,152,104),(0,160,96),(0,168,88),(0,176,80),(0,184,72),(0,192,64),(0,200,56),(0,208,48),(0,216,40),(0,224,32),(0,232,24),(0,240,16),(0,248,8),(0,255,0),(8,255,0),(16,255,0),(24,255,0),(32,255,0),(40,255,0),(48,255,0),(56,255,0),(64,255,0),(72,255,0),(80,255,0),(88,255,0),(96,255,0),(104,255,0),(112,255,0),(120,255,0),(128,255,0),(136,255,0),(144,255,0),(152,255,0),(160,255,0),(168,255,0),(176,255,0),(184,255,0),(192,255,0),(200,255,0),(208,255,0),(216,255,0),(224,255,0),(232,255,0),(240,255,0),(248,255,0),(255,255,0),(255,248,0),(255,240,0),(255,232,0),(255,224,0),(255,216,0),(255,208,0),(255,200,0),(255,192,0),(255,184,0),(255,176,0),(255,168,0),(255,160,0),(255,152,0),(255,144,0),(255,136,0),(255,128,0),(255,120,0),(255,112,0),(255,104,0),(255,96,0),(255,88,0),(255,80,0),(255,72,0),(255,64,0),(255,56,0),(255,48,0),(255,40,0),(255,32,0),(255,24,0),(255,16,0),(255,8,0),(255,0,0))

		allZero = "True"
		for item in self.itemList:
			x = float( item.getField( 'x' ) )
			y = float( item.getField( 'y' ) )
			
			if x != 0.0 or y != 0.0:
				allZero = False
				break
		
		if allZero:
			#This is check which version of kkit, b'cos anything below kkit8 didn't had xyz co-ordinates
			msgBox = QtGui.QMessageBox()
			msgBox.setText("The Layout module works for kkit version 8 or higher.")
			msgBox.setStandardButtons(QtGui.QMessageBox.Ok)
			msgBox.exec_()
			raise Widgetvisibility()
	
		else:		
			#Adding moose Object to scene to get scene bounding reaction for spacing the coordinates
			for item in self.itemList:
				pItem = Textitem(self,item.path)
				
				x = float(item.getField('x'))
				y = float(item.getField('y'))
				pItem.setPos(x,y)
				print "hhe",pItem.sceneBoundingRect()
				itemid = item.id
				
				#moosetext_dict[] is created to get the text's sceneBoundingRect
				self.moosetext_dict[itemid] = pItem 	

			scale_Cord = int(self.cordTransform(self.itemList,self.moosetext_dict))

			#Adding moose Object to scene and then adding to scene to view
			for item in self.itemList:
				pItem = Textitem(self,item.path)
				textColor = ""
				textBgcolor=""
				textColor = item.getField('xtree_textfg_req')
				textBgcolor = item.getField('xtree_fg_req')
				x = float(item.getField('x'))*(scale_Cord)
				y = float(item.getField('y'))*-(scale_Cord)
				pItem.setPos(x,y)
				pItem.setFlag(QtGui.QGraphicsItem.ItemIsMovable)
				#pItem.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges, 1)
				self.connect(pItem, QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),self.positionChange)
				self.connect(pItem, QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"), self.emitItemtoEditor)
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
				
				self.screen.addItem(pItem)

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
						src_id = item.id
						des_id = sobject
						self.lineCord(src,des,self.screen,item,self.object2line,self.lineItem_dict)
			
					for pobject in moose.Neutral(item.path).neighbours('prd', 0):
						src = self.moosetext_dict[pobject]
						des = self.moosetext_dict[item.id]
						src_id = pobject
						des_id = item.id
						self.lineCord(src,des,self.screen,item,self.object2line,self.lineItem_dict)
	
				#Added to substrate and product for reaction, we need to take enz also for the enzymetic Reaction	
				if(item.className == "Enzyme"):
					for pobject in moose.Neutral(item.path).neighbours('enz',0): 
						src = self.moosetext_dict[pobject]
						des = self.moosetext_dict[item.id]
						src_id = pobject
						des_id = item.id
						self.lineCord(src,des,self.screen,item,self.object2line,self.lineItem_dict)
			
			self.view = QtGui.QGraphicsView(self.screen,self)
			self.view.setScene(self.screen)
			grid.addWidget(self.view,0,0)
			
	def updateItemSlot(self, mooseObject):
			#In this case if the name is updated from the keyboard both in mooseobj and gui gets updation
	        for changedItem in (item for item in self.screen.items() if isinstance(item, Textitem) and mooseObject.id == item.mooseObj_.id):
	            break
	        changedItem.updateSlot()

	def keyPressEvent(self,event):
		#For Zooming
		for item in self.screen.items():
			if isinstance (item, Textitem):
				if((self.view.matrix().m11()<=1.0)and(self.view.matrix().m22() <=1.0)):
					item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, True)
				else:
					item.setFlag(QtGui.QGraphicsItem.ItemIgnoresTransformations, False)
					
		key = event.key()
		
		if key == QtCore.Qt.Key_A:
			self.view.resetMatrix()
	
		elif key == QtCore.Qt.Key_Plus:
			self.view.scale(1.1,1.1)

		elif key == QtCore.Qt.Key_Minus:
			self.view.scale(1/1.1,1/1.1)		
	
	def emitItemtoEditor(self,mooseObject):
		self.emit(QtCore.SIGNAL("itemDoubleClicked(PyQt_PyObject)"), mooseObject)		

	def positionChange(self,mooseObject):
		#If the item position changes, the corresponding arrow also changes here
		listItem = []
		for listItem in (v for k,v in self.object2line.items() if k.mooseObj_.id == mooseObject.id ):
			if len(listItem):
				for ql,va in listItem:
					srcdes = self.lineItem_dict[ql]
					arrow = self.calArrow(srcdes[0],srcdes[1])
					ql.setPolygon(arrow)
			break

	def lineCord(self,src,des,screen,source,object2line,lineItem_dict):
		if( (src == "") & (des == "") ):
			print "Source or destination is missing or incorrect"
		
		else:
			srcdes_list= [src,des]
			arrow = self.calArrow(src,des)
		
			if(source.className == "Reaction"):
				qgLineitem = self.screen.addPolygon(arrow,QtGui.QPen(QtCore.Qt.green, 1, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))

			elif(source.className == "Enzyme"):
				qgLineitem = self.screen.addPolygon(arrow,QtGui.QPen(QtCore.Qt.red, 1, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
			
			lineItem_dict[qgLineitem] = srcdes_list
			if src in object2line:
				object2line[ src ].append( ( qgLineitem, des) )
			else:
				 object2line[ src ] = []
				 object2line[ src ].append( ( qgLineitem, des) )
			
			if des in object2line:
				object2line[ des ].append( ( qgLineitem, src ) )
			else:
				object2line[ des ] = []
				object2line[ des ].append( ( qgLineitem, src) )
									
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
						
	def arrowHead(self,srcAngle,degree,lineSpoint):
		#arrowhead is calculated
		r = 8
		delta = math.radians(srcAngle) + math.radians(degree)
		width = math.sin(delta)*r
		height = math.cos(delta)*r
		srcXArr = lineSpoint.x() + width
		srcYArr = lineSpoint.y() + height
		return srcXArr,srcYArr
		
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

	def setupItem(self,mooseObject,itemlist):
		for child in mooseObject.children():
			childObj = moose.Neutral(child)
			if( (childObj.className == 'Molecule') | (childObj.className == 'Reaction') |(childObj.className == 'Enzyme') ):
				''' This is for eliminating enzyme complex'''
				if((moose.Neutral(childObj.parent).className) != 'Enzyme'):
					itemlist.append(childObj)
			self.setupItem(childObj,itemlist)

	def cordTransform(self,itemslist,mooseItemdict):
		#here alpha is calculated to multipy the coordinates with, so that the each items spreads out
		alpha = 0
		alpha1 = 0
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

if __name__ == "__main__":
	#app = QtGui.QApplication(sys.argv)
	dt = LayoutWidget()
	dt.show()
	sys.exit(app.exec_())


