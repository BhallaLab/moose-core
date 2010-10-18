import sys
import os
from PyQt4 import QtGui,QtCore,Qt
from operator import itemgetter, attrgetter
import math

import moose
'''
print os.getcwd()
os.chdir('/home/lab13/trunk/pymoose/gui')
sys.path.append('/home/lab13/trunk/pymoose/')

import moose
dir(moose)

c = moose.PyMooseBase.getContext()
c.loadG('/home/lab13/Genesis_file/acc59.g')
#c.loadG('/home/lab13/trunk/DEMOS/kholodenko/Kholodenko.g')

app = QtGui.QApplication(sys.argv)
'''

class Lineitem(QtGui.QGraphicsLineItem):
	def __init__(self,Xs,Ys,Xd,Yd,mooseitem):
		QtGui.QGraphicsLineItem.__init__(self)
		if(mooseitem.className == 'Reaction'): 
			self.setPen(Qt.QPen(Qt.Qt.green,1,Qt.Qt.SolidLine,Qt.Qt.RoundCap,Qt.Qt.RoundJoin))
		if(mooseitem.className == 'Enzyme'):
			self.setPen(Qt.QPen(Qt.Qt.red,1,Qt.Qt.SolidLine,Qt.Qt.RoundCap,Qt.Qt.RoundJoin))
		self.setLine(Xs,Ys,Xd,Yd)

class Textitem(QtGui.QGraphicsTextItem):
	def __init__(self,layoutwidget,path):
		self.mooseObj_ = moose.Neutral(path)
		self.layoutWidgetpt = layoutwidget
		QtGui.QGraphicsTextItem.__init__(self,self.mooseObj_.name)
		#self.setFlag(QtGui.QGraphicsItem.ItemIsMovable)


	def mouseDoubleClickEvent(self, event):
		self.emit(QtCore.SIGNAL("qgtextDoubleClick(PyQt_PyObject)"),self.mooseObj_)

	def updateSlot(self):
		if(self.mooseObj_.className == 'Enzyme'):
			textColor = "<html><body bgcolor='black'>"+self.mooseObj_.name+"</body></html>"
			self.setHtml(textColor)

		else:
			textColor = self.mooseObj_.getField('xtree_fg_req')
			self.layoutWidgetpt.colorCheck(self,self.mooseObj_,textColor,"background")

	
class LayoutWidget(QtGui.QWidget):
	def __init__(self,parent=None):
		QtGui.QWidget.__init__(self,parent)
		grid = QtGui.QGridLayout()
		self.setLayout(grid)
		
		self.rootObject = moose.Neutral('/kinetics')
		self.itemList = []
		self.setupItem(self.rootObject,self.itemList)
		self.moosetext_dict = {}

		self.colorMap = ((248,0,255),(240,0,255),(232,0,255),(224,0,255),(216,0,255),(208,0,255),(200,0,255),(192,0,255),(184,0,255),(176,0,255),(168,0,255),(160,0,255),(152,0,255),(144,0,255),(136,0,255),(128,0,255),(120,0,255),(112,0,255),(104,0,255),(96,0,255),(88,0,255),(80,0,255),(72,0,255),(64,0,255),(56,0,255),(48,0,255),(40,0,255),(32,0,255),(24,0,255),(16,0,255),(8,0,255),(0,0,255),(0,8,248),(0,16,240),(0,24,232),(0,32,224),(0,40,216),(0,48,208),(0,56,200),(0,64,192),(0,72,184),(0,80,176),(0,88,168),(0,96,160),(0,104,152),(0,112,144),(0,120,136),(0,128,128),(0,136,120),(0,144,112),(0,152,104),(0,160,96),(0,168,88),(0,176,80),(0,184,72),(0,192,64),(0,200,56),(0,208,48),(0,216,40),(0,224,32),(0,232,24),(0,240,16),(0,248,8),(0,255,0),(8,255,0),(16,255,0),(24,255,0),(32,255,0),(40,255,0),(48,255,0),(56,255,0),(64,255,0),(72,255,0),(80,255,0),(88,255,0),(96,255,0),(104,255,0),(112,255,0),(120,255,0),(128,255,0),(136,255,0),(144,255,0),(152,255,0),(160,255,0),(168,255,0),(176,255,0),(184,255,0),(192,255,0),(200,255,0),(208,255,0),(216,255,0),(224,255,0),(232,255,0),(240,255,0),(248,255,0),(255,255,0),(255,248,0),(255,240,0),(255,232,0),(255,224,0),(255,216,0),(255,208,0),(255,200,0),(255,192,0),(255,184,0),(255,176,0),(255,168,0),(255,160,0),(255,152,0),(255,144,0),(255,136,0),(255,128,0),(255,120,0),(255,112,0),(255,104,0),(255,96,0),(255,88,0),(255,80,0),(255,72,0),(255,64,0),(255,56,0),(255,48,0),(255,40,0),(255,32,0),(255,24,0),(255,16,0),(255,8,0),(255,0,0))

		self.screen = QtGui.QGraphicsScene(self)
		self.screen.setBackgroundBrush(QtGui.QColor(230,230,219,120))
		self.screen.setSceneRect(self.screen.itemsBoundingRect())
		
		#Adding moose Object to scene to get scene bounding reaction for spacing the coordinates
		for item in self.itemList:
			pItem = Textitem(self,item.path)
			x = float(item.getField('x'))
			y = float(item.getField('y'))
			pItem.setPos(x,y)
			test = pItem.sceneBoundingRect()
			itemid = item.id
			self.moosetext_dict[itemid] = pItem 	
		
		scale_Cord = int(self.cordTransform(self.itemList,self.moosetext_dict))
		
		#Adding moose Object to scene and then adding to scene to view
		for item in self.itemList:
			pItem = Textitem(self,item.path)
			textColor = ""
			textBgcolor=""
			textColor = item.getField('xtree_textfg_req')
			textBgcolor = item.getField('xtree_fg_req')
			x = float(item.getField('x'))*(scale_Cord+1)
			y = float(item.getField('y'))*-(scale_Cord+1)
			pItem.setPos(x,y)
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
				self.colorCheck(pItem,item,textColor,"foreground")
				self.colorCheck(pItem,item,textBgcolor,"background")
	
			self.screen.addItem(pItem)
		
		#connecting substrate,product to reaction and Enzyme
		for item in self.itemList:
			src = ""
			des = ""
			if(item.className != "Molecule"):
				for sobject in moose.Neutral(item.path).neighbours('sub', 0):
					src = self.moosetext_dict[item.id]
					des = self.moosetext_dict[sobject]
					source = item
					self.lineCord(src,des,self.screen,item)		
		
				for pobject in moose.Neutral(item.path).neighbours('prd', 0):
					src = self.moosetext_dict[pobject]
					des = self.moosetext_dict[item.id]
					source = moose.Neutral(pobject)
					self.lineCord(src,des,self.screen,item)

			#Added to substrate and product for reaction, we need to take enz also for the enzymetic Reaction		
			if(item.className == "Enzyme"):
				for pobject in moose.Neutral(item.path).neighbours('enz',0): 
					src = self.moosetext_dict[pobject]
					des = self.moosetext_dict[item.id]
					source = moose.Neutral(pobject)
					self.lineCord(src,des,self.screen,item)
		test = self.screen.itemsBoundingRect()
		print ">",test,test.center()
		view = QtGui.QGraphicsView(self.screen,self)
		view.setScene(self.screen)
		#view.ensureVisible(50,50)
		grid.addWidget(view,0,0)

	def updateItemSlot(self, mooseObject):
	        for changedItem in (item for item in self.screen.items() if isinstance(item, Textitem) and mooseObject.id == item.mooseObj_.id):
	            break
	        changedItem.updateSlot()
		
	def emitItemtoEditor(self,mooseObject):
		self.emit(QtCore.SIGNAL("itemDoubleClicked(PyQt_PyObject)"), mooseObject)		

	def lineCord(self,src,des,screen,source):
		if( (src == "") & (des == "") ):
			print "Source or destination is missing or incorrect"
		else:
			srcRect = src.sceneBoundingRect()
			sx = srcRect.center().x()
			sy = srcRect.center().y()
			sxMid = 1.15*(srcRect.right()-srcRect.left())/2
			syMid = 1.15*(srcRect.bottom()-srcRect.top())/2
			
			''' destination'''
			desRect = des.sceneBoundingRect()
			dx = desRect.center().x()
			dy = desRect.center().y()
			dxMid = 1.15*(desRect.right()-desRect.left())/2
			dyMid = 1.15*(desRect.bottom()-desRect.top())/2
	
			t = math.atan2((sxMid*-(dy-sy)),syMid*(dx-sx))
			srcXf = sx + (sxMid*math.cos(t))
			srcYf = sy - (syMid*math.sin(t))
	
			t1 = t + math.pi
			desXf = dx + (dxMid*math.cos(t1))
			desYf = dy - (dyMid*math.sin(t1))

			if((srcXf != "") & (desXf != "") & (srcYf != "") & (desYf != "") ):
				if(((abs(srcXf-desXf)) > 10 ) |((abs(srcYf-desYf)) > 10 ) ):
					pLine = Lineitem(srcXf,srcYf,desXf,desYf,source)
					self.screen.addItem(pLine)

			 		#Calculate arrows head
					r = 10
					theta = math.atan2(-(dy-sy),(dx-sx))
					delta = theta + math.radians(60)
					width = math.sin(delta)*r
					height = math.cos(delta)*r
					srcXArr = srcXf + width
					srcYArr = srcYf + height
					pLine = Lineitem(srcXf,srcYf,srcXArr,srcYArr,source)
					self.screen.addItem(pLine)
	
					delta = theta + math.radians(120)
					width = math.sin(delta)*r
					height = math.cos(delta)*r
					srcXArr = srcXf + width
					srcYArr = srcYf + height
					pLine = Lineitem(srcXf,srcYf,srcXArr,srcYArr,source)
					self.screen.addItem(pLine)

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

	def cordTransform(self,itemslist,mooseDict):
		alpha =0
		alpha1 =0
		for m in range(len(itemslist)):
			for n in range(len(itemslist)):
				if(m != n):
					src = itemslist[m]
					self.pitemsrc = mooseDict[src.id]
					src.Scenebounding = self.pitemsrc.sceneBoundingRect()
					srcX = src.Scenebounding.x()
					srcY = src.Scenebounding.y()
					srcW = src.Scenebounding.right()-src.Scenebounding.left()
					srcH = src.Scenebounding.bottom()-src.Scenebounding.top()

					des = itemslist[n]
					self.pitemdes = mooseDict[des.id]
					des.Scenebounding = self.pitemdes.sceneBoundingRect()
					desX = des.Scenebounding.x()
					desY = des.Scenebounding.y()
					desW = des.Scenebounding.right()-des.Scenebounding.left()
					desH = des.Scenebounding.bottom()-des.Scenebounding.top()

					#Checking for overlap of items
					alpha = self.overLap(float(srcX),float(srcY),srcW,srcH,float(desX),float(desY),desW,desH)	
					if(alpha1 < alpha): alpha1 = alpha
					else:	pass
			
				else: 
					pass
		return(alpha1)

	def rectBoundry(self,x0,y0,wi,hi,px,py,wj,hj):
		if( (px >= x0) and (px <= x0 +wi) and (py >= y0) and (py <=y0+hj) ):		return 1
		else:		return 0

	def cal_Alpha(self,XI0,YI0,WI,HI,XI1,YI1):
		sfx = 0
		sfy = 0
		if(XI1-XI0 != 0):		sfx = ( float(WI)/abs(XI1-XI0) )
		if(YI1-YI0 != 0):		sfy = ( float(HI)/abs(YI1-YI0) )

		if((sfx != 0) and (sfy != 0)):
			if( sfx < sfy):		return sfx
			elif (sfy < sfx):	return sfy
			else:			return 0
		elif (sfx == 0): return sfy
		elif (sfy == 0): return sfx
		else:		 return 0

	def overLap(self,xi0,yi0,wi,hi,xj0,yj0,wj,hj):
		fact = 0
		previous = 0
		present_alp = 0
		if(fact == 0):
			fact = self.rectBoundry(xi0,yi0,wi,hi,xj0,yj0,wj,hj)
			if(fact == 0):
				fact = self.rectBoundry(xi0,yi0,wi,hi,xj0,yj0+hj,wj,hj)
				if(fact == 0):
					fact = self.rectBoundry(xi0,yi0,wi,hi,xj0+wj,yj0,wj,hj)
					if(fact == 0):
						fact = self.rectBoundry(xi0,yi0,wi,hi,xj0+wj,yj0,wj,hj)
		if(fact == 1):
			present_alp = self.cal_Alpha(xi0,yi0,wi,hi,xj0,yj0)

		if(previous < present_alp): previous = present_alp

		return present_alp

if __name__ == "__main__":
	#app = QtGui.QApplication(sys.argv)
	dt = LayoutWidget()
	dt.show()
	sys.exit(app.exec_())


