import moose
from kkitQGraphics import * 
#PoolItem, ReacItem,EnzItem,CplxItem,ComptItem
from PyQt4 import QtCore,QtGui


def NewObject(EditWB,view,modelRoot,newString,mapToscene,createdItem):
    ''' Rules: Compartment should be created before created any mooseObject and they are placed side by side.
               All object Pool,reaction,enz are assocaited to a compartment and not outside.
               When Enz created its complex is alsocreated along with default value user may edit its value.
    '''
    Item = ''
    num = ''
    print "modelRoot",modelRoot
    createdItem = createdItem
    if newString in createdItem:
        num = createdItem[newString]
        createdItem[newString] += 1
    else:
        createdItem[newString] = 1
    if newString == 'CubeMesh':
        #compartment = checkExist(modelRoot,newString,createdItem)
        if not num is None:
            compartment = moose.CubeMesh(modelRoot+'/compartment'+str(num) )
        else:
            compartment = moose.CubeMesh(modelRoot+'/compartment')
        compartment.volume = 1e-15
        mesh = moose.element(compartment.path+'/mesh') 
        Item = ComptItem(EditWB,0,0,100,100,mesh)
        Item.moveBy(mapToscene.x(),mapToscene.y())
        Item.setPen(QtGui.QPen(Qt.QColor(66,66,66,100), 5, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
        view.emit(QtCore.SIGNAL("dropped"),compartment)

    elif newString == 'CylMesh':
        #compartment = checkExist(modelRoot,newString,createdItem)
        if not num is None:
            compartment = moose.CylMesh(modelRoot+'/CylCompartment'+str(num) )
        else:
            compartment = moose.CylMesh(modelRoot+'/CylCompartment')
        compartment.volume = 1e-15
        mesh = moose.element(compartment.path+'/mesh') 
        Item = ComptItem(EditWB,0,0,100,100,mesh)
        Item.moveBy(mapToscene.x(),mapToscene.y())
        view.emit(QtCore.SIGNAL("dropped"),compartment)

    elif newString == 'Reac':
        if not num is None:
            reaction = moose.Reac(modelRoot+'/reac'+str(num))
        else:
            reaction = moose.Reac(modelRoot+'/reac')
        Item = ReacItem(reaction)
        Item.setDisplayProperties(mapToscene.x(),mapToscene.y(),'','')
        view.emit(QtCore.SIGNAL("dropped"),reaction)
    elif newString == 'Enz':
        if not num is None:
            enzyme = moose.Enz(modelRoot+'/enz'+str(num))
        else:
            enzyme = moose.Enz(modelRoot+'/enz')
        Item = EnzItem(enzyme)
        Item.setDisplayProperties(mapToscene.x(),mapToscene.y(),QtGui.QColor('blue'),'')
        cplxPool = moose.Pool(enzyme.path+'/'+enzyme.name+'_cplx')
        cplxItem = CplxItem(cplxPool,Item)
        cplxItem.setDisplayProperties(mapToscene.x(),mapToscene.y(),'','')
        view.emit(QtCore.SIGNAL("dropped"),enzyme)

    elif newString == 'MMenz':
        if not num is None:
            MMEnzyme = moose.MMenz(modelRoot+'/MMEnz'+str(num))
        else:
            MMEnzyme = moose.MMenz(modelRoot+'/MMEnz')
        Item = MMEnzItem(MMEnzyme)
        Item.setDisplayProperties(mapToscene.x(),mapToscene.y(),QtGui.QColor('green'),'')
        view.emit(QtCore.SIGNAL("dropped"),MMEnzyme)
    return Item

def checkExist(modelRoot,mooseClass,mooseDict):
    mObj = ''
    print mooseDict
    if mooseClass == 'CubeMesh' or mooseClass == 'CylMesh':
        if not moose.exists(modelRoot+'/compartment') and not mooseDict.has_key('Mesh'):
            mObj = moose.CubeMesh( modelRoot+'/compartment' )
            mooseDict['Mesh'] = 1
        else:
            num = mooseDict['Mesh']
            mObj = moose.CubeMesh( modelRoot+'/compartment'+str(num) )
            mooseDict['Mesh'] +=1
        
    elif mooseClass == 'Reac' or mooseClass == 'Enz' or mooseClass == 'MMenz':
        
        '''
        if not moose.exists(modelRoot+'/Reac'):
            #TODO: NEED TO ACCESS P R E WITH RESPECT TO COMPARTMENT
            
            RE = moose.Neutral('Mesh path').getNeighbors('remeshReacs')
        ''' 
    return mObj

