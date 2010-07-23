import sys
import os
from PyQt4.QtCore import *
from PyQt4.QtGui import *
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
c.loadG('/home/lab13/Genesis_file/Anno_acc8.g')


app = QtGui.QApplication(sys.argv)
'''
font = QtGui.QFont()
fm = QtGui.QFontMetrics(font)

class Textitem(QtGui.QGraphicsTextItem):
	def __init__(self,path):
		tname = path[(path.rfind('/'))+1:len(path)]
		QtGui.QGraphicsTextItem.__init__(self,tname)
		self.sname=tname
		self.spath=path
		
	def mouseDoubleClickEvent(self, event):
		pass
		#print "I have double clicked " + self.spath
		
		
class Lineitem(QtGui.QGraphicsLineItem):
	def __init__(self,Xs,Ys,Xd,Yd,root):
		QtGui.QGraphicsLineItem.__init__(self)
		if(moose.Neutral(root).className == 'Reaction'): 
			self.setPen(Qt.QPen(Qt.Qt.green,1))
		if(moose.Neutral(root).className == 'Enzyme'):
			self.setPen(Qt.QPen(Qt.Qt.red,1))
		self.setLine(Xs,Ys,Xd,Yd)

class xycord_cl:
	''' This Class keeps record of name, x & y co-ordinates and C->color of the model '''
	def __init__(self,nme,x,y,c):
		self.name=nme
		self.x=x
		self.y=y
		self.c=c
	def getName(self):		return self.name
	def getX(self):			return self.x
	def getY(self):			return self.y
	def getC(self):			return self.c
	def setX(self,x):		self.x = x
	def setY(self,y):		self.y = y
	def setC(self,c):		self.c = c

class arrow_cl:
	''' This Class keeps record for connecting arrows name,dir -> incoming (1) or outgoing (0)'''
	def __init__(self,nam,d):
		self.name = nam
		self.dir = d
		
	def getName(self):	return self.name
	def getDir(self):	return self.dir

	def setD(self,d):	self.dir = d

def fallinsiderect(x0,y0,wi,px,py, wj):
	sfx = 0
	sfy = 0
	if( (px >= x0) and (px <= x0 +wi) and (py >= y0) and (py <=y0+fm.height()) ):		return 1
	else:		return 0

def cal_alpha(XI0,YI0,WI,XI1,YI1):
		sfx = 0
		sfy = 0
		if(XI1-XI0 != 0):		sfx = ( float(WI)/abs(XI1-XI0) )
		if(YI1-YI0 != 0):		sfy = ( float(fm.height())/abs(YI1-YI0) )

		if((sfx != 0) and (sfy != 0)):
			if( sfx < sfy):		return sfx
			elif (sfy < sfx):	return sfy
			else:			return 0
		elif (sfx == 0): return sfy
		elif (sfy == 0): return sfx
		else:		 return 0

def overlap(xi0,yi0,wi,xj0,yj0,wj):
	fact = 0
	previous = 0
	present_alp = 0
	if(fact == 0):
		fact = fallinsiderect(xi0,yi0,wi,xj0,yj0, wj)
		if(fact == 0):
			fact = fallinsiderect(xi0,yi0,wi,xj0,yj0 + fm.height(), wj)
			if(fact == 0):
				fact = fallinsiderect(xi0,yi0,wi,xj0 + wj,yj0, wj)
				if(fact == 0):
					fact = fallinsiderect(xi0,yi0,wi,xj0 + wj,yj0, wj)
	if(fact == 1):
		present_alp = cal_alpha(xi0,yi0,wi,xj0,yj0)
	if(previous < present_alp): previous = present_alp

	return present_alp

def _genesis_xyc():
	xycord_list = []
	SubPrd_dict = {}
	Mol_dict = {}
	alpha =0
	alpha1 =0
	
	for object in moose.Neutral('/kinetics').children():
		if( (moose.Neutral(object)).className == 'KinCompt'):
			for compobj in moose.Neutral(object.path()).children():
				xycord_list.append(xycord_cl(compobj.path(),moose.Neutral(compobj).getField('x'),moose.Neutral(compobj).getField	('y'),moose.Neutral(compobj).getField('xtree_textfg_req')))
			
				if( (moose.Neutral(compobj)).className == 'Molecule'):
					Mol_list = []
					for reactotal in moose.Neutral(compobj).neighbours('reac'): Mol_list +=[(arrow_cl(reactotal.path(),0))]
					for enzprdtotal in moose.Neutral(compobj).neighbours('prd'): Mol_list +=[(arrow_cl(enzprdtotal.path(),0))]
					if(len(Mol_list) != 0): Mol_dict[compobj.path()] = Mol_list
	
	
					for enz in moose.Neutral(compobj.path()).children():
						#This will get path,x,y cordinates for the layout in xycord_list
						xycord_list.append(xycord_cl(enz.path(),moose.Neutral(enz).getField('x'),moose.Neutral(enz).getField	('y'),moose.Neutral(enz).getField('xtree_textfg_req')))
	 					#This will provide subtrate and product for the connection in SubPrd_dict
						slist = []
						for sobject in moose.Neutral(enz).neighbours('sub',0): slist+=[arrow_cl(sobject.path(),1)]
						for pobject in moose.Neutral(enz).neighbours('prd',0): slist+=[arrow_cl(pobject.path(),0)]
						for pobject in moose.Neutral(enz).neighbours('enz',0): slist+=[arrow_cl(pobject.path(),0)]
						if ( len(slist) != 0): 	        			SubPrd_dict[enz.path()] = slist
		
				elif( (moose.Neutral(compobj)).className == 'Reaction'):
					slist = []
					for sobject in moose.Neutral(compobj).neighbours('sub', 0):  slist+=[arrow_cl(sobject.path(),1)]
					for pobject in moose.Neutral(compobj).neighbours('prd', 0):  slist+=[arrow_cl(pobject.path(),0)]
					if ( len(slist) != 0): SubPrd_dict[compobj.path()] = slist
					
						
		elif( (moose.Neutral(object)).className == 'Neutral'):
			pass
		else:
			xycord_list.append(xycord_cl(object.path(),moose.Neutral(object).getField('x'),moose.Neutral(object).getField('y'),moose.Neutral(object).getField('xtree_textfg_req')))
			if( (moose.Neutral(object)).className == 'Molecule'):
				Mol_list = []
				for reactotal in moose.Neutral(object).neighbours('reac'): Mol_list +=[(arrow_cl(reactotal.path(),0))]
				for enzprdtotal in moose.Neutral(object).neighbours('prd'): Mol_list +=[(arrow_cl(enzprdtotal.path(),0))]
				if(len(Mol_list) != 0): Mol_dict[object.path()] = Mol_list
	
	
				sumlist = []
				for S_enz in moose.Neutral(object.path()).children():
						xycord_list.append(xycord_cl(S_enz.path(),moose.Neutral(S_enz).getField('x'),moose.Neutral(S_enz).getField('y'),moose.Neutral(S_enz).getField('xtree_textfg_req')))
						#This will provide subtrate and product for the connection in SubPrd_dict
						slist = []
						for sobject in moose.Neutral(S_enz).neighbours('sub',0): slist+=[arrow_cl(sobject.path(),1)]
						for pobject in moose.Neutral(S_enz).neighbours('prd',0): slist+=[arrow_cl(pobject.path(),0)]
						for pobject in moose.Neutral(S_enz).neighbours('enz',0): slist+=[arrow_cl(pobject.path(),0)]
						if ( len(slist) != 0): 	                                 SubPrd_dict[S_enz.path()] = slist
			elif( (moose.Neutral(object)).className == 'Reaction'):
					slist = []
					for sobject in moose.Neutral(object).neighbours('sub', 0): slist+=[arrow_cl(sobject.path(),1)]
					for pobject in moose.Neutral(object).neighbours('prd', 0): slist+=[arrow_cl(pobject.path(),0)]
					if ( len(slist) != 0): 					   SubPrd_dict[object.path()] = slist
				

	#Checking which co-ordinates collide with each other
	for m in range(len(xycord_list)):
		for n in range (len(xycord_list)):
			if(m != n):
				alpha = overlap(float(xycord_list[m].getX()),float(xycord_list[m].getY()),fm.width(xycord_list[m].getName()[(xycord_list[m].getName().rfind('/'))+1:len(xycord_list[m].getName())]),float(xycord_list[n].getX()),float(xycord_list[n].getY()),fm.width(xycord_list[n].getName()[(xycord_list[n].getName().rfind('/'))+1:len(xycord_list[n].getName())]))       
				if(alpha1 < alpha): alpha1 = alpha
			else: 
				pass

	
	#''' Multiply factor to space out the molecule,reaction, enzyme for display'''
	for n in range(len(xycord_list)):
		xycord_list[n].setX(float(xycord_list[n].getX())*alpha1)
		xycord_list[n].setY(float(xycord_list[n].getY())*alpha1)

	#I wanted all the value in the graph should be positive value so checking the max x and y value and adding
	max_X = 0.0
	max_Y = 0.0
	for n in range(len(xycord_list)):
		
		if(max_X > xycord_list[n].getX()):
			max_X = xycord_list[n].getX()

		if(max_Y > xycord_list[n].getY()):
			max_Y = xycord_list[n].getY()

	for n in range(len(xycord_list)):
		xycord_list[n].setX(float(xycord_list[n].getX())+abs(max_X))
		xycord_list[n].setY(float(xycord_list[n].getY())+abs(max_Y))

	return xycord_list,SubPrd_dict,Mol_dict

class Screen(QtGui.QWidget):
	def __init__(self,parent=None):
		QtGui.QWidget.__init__(self,parent)
		grid = QtGui.QGridLayout()
		self.setLayout(grid)
		self.scene = QtGui.QGraphicsScene(self)
		self.scene.setBackgroundBrush(QtGui.QColor(230,230,219,120))
		xycord_list = []
		SubPrd_dict = {}
		Mol_dict = {}
		(xycord_list,SubPrd_dict,Mol_dict)  = _genesis_xyc()

		self.DisplayText(xycord_list)
		self.DisplayLine(xycord_list,SubPrd_dict,Mol_dict)

		view = QtGui.QGraphicsView(self.scene,self)
		view.setScene(self.scene)
		grid.addWidget(view,0,0)

	def DisplayText(self,xycord_list):		
		#Add text items to screen
		for item in xycord_list:
			pItem = Textitem(item.getName())
			pItem.setPos(item.getX(),item.getY())
			pItem.setDefaultTextColor(QColor(item.getC()))
			self.scene.addItem(pItem)

	def DisplayLine(self,xycord_list,SubPrd_dict,Mol_dict):
		''' Drawing the lines for connectivity l,m of SubPrd_dict should be same as o,p of Mol_dict 
				l	p
				m	o	
		'''
		number = 0
                
		for l,m in SubPrd_dict.iteritems():
			number = number +1
			for xy_items_sp in xycord_list:
				if(xy_items_sp.getName() == l):
					break
			for sp_items in m:
				for o,p in Mol_dict.iteritems():
					if(o == sp_items.getName()):
						for xy_items_ml in xycord_list:
							if(xy_items_ml.getName() == o):
								break
						found = 0
						for mol_items in p:
							if(mol_items.getName() == l):
								found = 1
								#Source
								xs = xy_items_sp.getX()
								ys = xy_items_sp.getY()
								ao = ( 1.15 * (float(fm.width(l[(l.rfind('/'))+1:len(l)]))) )/ 2
								bo = (1.15*fm.height())/2
								xcs = xs+ao
								ycs = ys+bo
								#Destination
								xd = xy_items_ml.getX()
								yd = xy_items_ml.getY()
								a1 = (1.15 * (float(fm.width(o[(o.rfind('/'))+1:len(o)])) ))/2
								b1 = (1.15*fm.height())/2
								xcd = xd+a1
								ycd = yd+b1
								#calcualte theta,t and x and y final co-ordinates
								theta = math.atan2(-(ycd-ycs),(xcd-xcs))
								
								t = math.atan2((ao*-(ycd-ycs)),bo*(xcd-xcs))
								xsf = xcs + (ao*math.cos(t))
								ysf = ycs - (bo*math.sin(t))

								td = t + math.pi
								xdf = xcd + (a1*math.cos(td))
								ydf = ycd - (b1*math.sin(td))

								if((xsf != "") & (xdf != "") & (ysf != "") & (ydf != "") ):
									if((abs(xsf-xdf) > 10 ) | (abs(ysf-ydf) > 10 ) ):
										pLine = Lineitem(xsf,ysf,xdf,ydf,l)
										self.scene.addItem(pLine)
	
										if(sp_items.getDir() == 1):
											r = 10
											delta = theta + math.radians(60)
											width = math.sin(delta)*r
											height = math.cos(delta)*r
											xsa = xsf + width
											ysa = ysf + height
											pLine = Lineitem(xsf,ysf,xsa,ysa,l)
											self.scene.addItem(pLine)
											delta1 = theta + math.radians(120)
											width = math.sin(delta1)*r
											height = math.cos(delta1)*r
											xda = xsf + width
											yda = ysf + height
											pLine = Lineitem(xsf,ysf,xda,yda,l)
											self.scene.addItem(pLine)
										elif(sp_items.getDir() == 0):
											delta = theta + math.radians(180)+math.radians(60)
											r = 10
											width = math.sin(delta)*r
											height = math.cos(delta)*r
											xsa = xdf + width
											ysa = ydf + height
											pLine = Lineitem(xdf,ydf,xsa,ysa,l)
											self.scene.addItem(pLine)
											delta1 = theta + math.radians(180) + math.radians(120)
											width = math.sin(delta1)*r
											height = math.cos(delta1)*r
											xsa = xdf + width
											ysa = ydf + height
											pLine = Lineitem(xdf,ydf,xsa,ysa,l)
											self.scene.addItem(pLine)
	
								        else: pass
						if(found == 0): pass
 

if __name__ == "__main__":
	#app = QtGui.QApplication(sys.argv)
	dt = Screen()
	dt.show()
	sys.exit(app.exec_())

