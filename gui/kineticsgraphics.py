# Filename: kineticsgraphics.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Copyright (C) 2010 Subhasis Ray, all rights reserved.
# Created: Sun Sep 23 21:32:21 2012 (+0530)
# Version: 
# Last-Updated: Mon Sep 24 01:41:16 2012 (+0530)
#           By: subha
#     Update #: 2164
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# GraphicsItems for displaying chemical kintics elements.
# 
# 

# Change log:
# 
# 
# 

# Code:

from PyQt4 import QtGui, QtCore
from kineticsutils import displayinfo
import moose

class KineticsDisplayItem(QtGui.QGraphicsWidget):
    """Base class for display elemenets in kinetics layout"""
    def __init__(self, mooseObject, parent=None):
        QtGui.QGraphicsObject.__init__(self, parent)
        self.mobj = mooseObject
        self.gobj = None

    def setDisplayProperties(self, dinfo):
        self.setGeometry(dinfo.x, dinfo.y)        
    
class PoolItem(KineticsDisplayItem):
    """Class for displaying pools. Uses a QGraphicsSimpleTextItem to
    display the name."""    
    fontMetrics = None
    def __init__(self, *args, **kwargs):
        KineticsDisplayItem.__init__(self, *args, **kwargs)
        self.bg = QtGui.QGraphicsRectItem(self)
        self.gobj = QtGui.QGraphicsSimpleTextItem(self.mobj.name, self.bg)        
        if not PoolItem.fontMetrics:
            PoolItem.fontMetrics = QtGui.QFontMetrics(self.gobj.font())
        self.bg.setRect(0, 
                        0, 
                        self.gobj.boundingRect().width()
                        +PoolItem.fontMetrics.width('  '), 
                        self.gobj.boundingRect().height())
        self.gobj.setPos(PoolItem.fontMetrics.width(' '), 0)

    def setDisplayProperties(self, dinfo):
        """Set the display properties of this item."""
        self.setGeometry(dinfo.x, dinfo.y, 
                         self.bg.boundingRect().width(), 
                         self.bg.boundingRect().height())
        self.gobj.setPen(QtGui.QPen(QtGui.QBrush(dinfo.fc)))
        self.gobj.setBrush(QtGui.QBrush(dinfo.fc))
        self.bg.setBrush(QtGui.QBrush(dinfo.bc))
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable
                     +QtGui.QGraphicsItem.ItemIsMovable)
        if QtCore.QT_VERSION >= 0x040600:
            self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)

class ReacItem(KineticsDisplayItem):
    defaultWidth = 20
    defaultHeight = 10
    def __init__(self, *args, **kwargs):
        KineticsDisplayItem.__init__(self, *args, **kwargs)
        self.gobj = QtGui.QGraphicsEllipseItem(0, 0, 
                                               ReacItem.defaultWidth, 
                                               ReacItem.defaultHeight, self)
        
    def setDisplayProperties(self, dinfo):
        """Set the display properties of this item."""
        self.setGeometry(dinfo.x, dinfo.y, 
                         self.gobj.boundingRect().width(), 
                         self.gobj.boundingRect().height())
        self.gobj.setPen(QtGui.QPen(QtGui.QBrush(dinfo.fc)))
        self.gobj.setBrush(QtGui.QBrush(dinfo.bc))
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable
                     +QtGui.QGraphicsItem.ItemIsMovable)
        if QtCore.QT_VERSION >= 0x040600:
            self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)

class EnzItem(KineticsDisplayItem):
    defaultWidth = 20
    defaultHeight = 10    
    def __init__(self, *args, **kwargs):
        KineticsDisplayItem.__init__(self, *args, **kwargs)
        self.gobj = QtGui.QGraphicsRectItem(0, 0, 
                                            EnzItem.defaultWidth, 
                                            EnzItem.defaultHeight, self)
        
    def setDisplayProperties(self, dinfo):
        """Set the display properties of this item."""
        self.setGeometry(dinfo.x, dinfo.y, 
                         self.gobj.boundingRect().width(), 
                         self.gobj.boundingRect().height())
        self.gobj.setPen(QtGui.QPen(QtGui.QBrush(dinfo.fc)))
        self.gobj.setBrush(QtGui.QBrush(dinfo.bc))
        self.setFlag(QtGui.QGraphicsItem.ItemIsSelectable
                     +QtGui.QGraphicsItem.ItemIsMovable)
        if QtCore.QT_VERSION >= 0x040600:
            self.setFlag(QtGui.QGraphicsItem.ItemSendsGeometryChanges)
        print self.mobj.path, self.pos(), self.boundingRect(), dinfo.bc.name()
        

import sys
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    a = moose.Pool('pool')
    b = moose.Reac('reac')
    c = moose.Enz('enzyme')
    gview = QtGui.QGraphicsView()
    scene = QtGui.QGraphicsScene(gview)
    item = PoolItem(a)
    dinfo = displayinfo(5, 5, QtGui.QColor('red'), QtGui.QColor('blue'))
    item.setDisplayProperties(dinfo)
    reacItem = ReacItem(b)
    reacItem.setDisplayProperties(displayinfo(50, 5, QtGui.QColor('yellow'), QtGui.QColor('green')))
    enzItem = EnzItem(c)
    enzItem.setDisplayProperties(displayinfo(100, 10, QtGui.QColor('blue'), QtGui.QColor('yellow')))
    scene.addItem(item)
    scene.addItem(reacItem)
    scene.addItem(enzItem)
    gview.setScene(scene)
    print 'Position', reacItem.pos()
    gview.show()
    sys.exit(app.exec_())
