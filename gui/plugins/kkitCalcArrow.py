from PyQt4 import QtGui,QtCore,Qt
import math
from kkitQGraphics import PoolItem, ReacItem,EnzItem,CplxItem,ComptItem

''' One to need to pass the source, destination and endtype for drawing the arrow between 2 object \
    endtype is to check if needs arrow head (arrowhead for product and sumtotal)
'''
def calcArrow(src,des,endtype,itemignoreZooming,iconScale):
    ''' if PoolItem then boundingrect should be background rather than graphicsobject '''
    srcobj = src.gobj
    desobj = des.gobj
    if isinstance(src,PoolItem):
        srcobj = src.bg
    if isinstance(des,PoolItem):
        desobj = des.bg
            
    if itemignoreZooming:
        srcRect = self.recalcSceneBoundingRect(srcobj)
        desRect = self.recalcSceneBoundingRect(desobj)
    else:
        srcRect = srcobj.sceneBoundingRect()
        desRect = desobj.sceneBoundingRect()
    arrow = QtGui.QPolygonF()
    if srcRect.intersects(desRect):                
        ''' This is created for getting a emptyline reference \
            because 'lineCord' function keeps a reference between qgraphicsline and its src and des
        '''
        arrow.append(QtCore.QPointF(0,0))
        arrow.append(QtCore.QPointF(0,0))
        return arrow
    tmpLine = QtCore.QLineF(srcRect.center().x(),
                                    srcRect.center().y(),
                                    desRect.center().x(),
                                    desRect.center().y())
    srcIntersects, lineSrcPoint = calcLineRectIntersection(srcRect, tmpLine)
    destIntersects, lineDestPoint = calcLineRectIntersection(desRect, tmpLine)
    if not srcIntersects:
        print 'Source does not intersect line. Arrow points:', lineSrcPoint, src.mobj[0].name, src.mobj[0].class_
    if not destIntersects:
        print 'Dest does not intersect line. Arrow points:', lineDestPoint,  des.mobj[0].name, des.mobj[0].class_

    '''src and des are connected with line co-ordinates
       Arrow head is drawned if the distance between src and des line is >8 just for clean appeareance
    '''
    if (abs(lineSrcPoint.x()-lineDestPoint.x()) > 8 or abs(lineSrcPoint.y()-lineDestPoint.y())>8):
        srcAngle = tmpLine.angle()
        if endtype == 'p':
            ''' Arrow head for Destination is calculated'''
            arrow.append(lineSrcPoint)
            arrow.append(lineDestPoint)
            degree = -60
            srcXArr1,srcYArr1= arrowHead(srcAngle,degree,lineDestPoint,iconScale)
            arrow.append(QtCore.QPointF(srcXArr1,srcYArr1))
            arrow.append(QtCore.QPointF(lineDestPoint.x(),lineDestPoint.y()))
                
            degree = -120
            srcXArr2,srcYArr2 = arrowHead(srcAngle,degree,lineDestPoint,iconScale)
            arrow.append(QtCore.QPointF(srcXArr2,srcYArr2))                    
            arrow.append(QtCore.QPointF(lineDestPoint.x(),lineDestPoint.y()))
 
        elif endtype == 'st':
            ''' Arrow head for Source is calculated'''
            arrow.append(lineDestPoint)
            arrow.append(lineSrcPoint)
            degree = 60
            srcXArr2,srcYArr2 = arrowHead(srcAngle,degree,lineSrcPoint,iconScale)
            arrow.append(QtCore.QPointF(srcXArr2,srcYArr2))                    
            arrow.append(QtCore.QPointF(lineSrcPoint.x(),lineSrcPoint.y()))

            degree = 120
            srcXArr1,srcYArr1= arrowHead(srcAngle,degree,lineSrcPoint,iconScale)
            arrow.append(QtCore.QPointF(srcXArr1,srcYArr1))
            arrow.append(QtCore.QPointF(lineSrcPoint.x(),lineSrcPoint.y()))

        else:
            arrow.append(lineSrcPoint)
            arrow.append(lineDestPoint)
    return arrow

def calcLineRectIntersection(rect, centerLine):
    '''      checking which side of rectangle intersect with centerLine \
        Here the 1. a. intersect point between center and 4 sides of src and \
                    b. intersect point between center and 4 sides of des and \
                     to draw a line connecting for src & des
                 2. angle for src for the arrow head calculation is returned
    '''
    x = rect.x()
    y = rect.y()
    w = rect.width()
    h = rect.height()
    borders = [(x,y,x+w,y),
                   (x+w,y,x+w,y+h),
                   (x+w,y+h,x,y+h),
                   (x,y+h,x,y)]
    intersectionPoint = QtCore.QPointF()
    intersects = False
    for lineEnds in borders:
        line = QtCore.QLineF(*lineEnds)
        intersectType = centerLine.intersect(line, intersectionPoint)
        if intersectType == centerLine.BoundedIntersection:
            intersects = True
            break
    return (intersects, intersectionPoint)

def arrowHead(srcAngle,degree,lineSpoint,iconScale):
    '''  arrow head is calculated '''
    r = 8*iconScale
    delta = math.radians(srcAngle) + math.radians(degree)
    width = math.sin(delta)*r
    height = math.cos(delta)*r
    srcXArr = (lineSpoint.x() + width)
    srcYArr = (lineSpoint.y() + height)
    return srcXArr,srcYArr
