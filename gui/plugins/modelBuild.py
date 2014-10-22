import moose
from kkitQGraphics import * 
#rom kkitOrdinateUtil import *
import PyQt4

def updateCompartmentSize(qGraCompt):
    childBoundingRect = qGraCompt.childrenBoundingRect()
    comptBoundingRect = qGraCompt.boundingRect()
    rectcompt = comptBoundingRect.united(childBoundingRect)
    comptPen = qGraCompt.pen()
    comptWidth =  5
    comptPen.setWidth(comptWidth)
    qGraCompt.setPen(comptPen)
    if not comptBoundingRect.contains(childBoundingRect):
        qGraCompt.setRect(rectcompt.x()-comptWidth,rectcompt.y()-comptWidth,rectcompt.width()+(comptWidth*2),rectcompt.height()+(comptWidth*2))

# def checkCreate(string,num,itemAt,qGraCompt,modelRoot,scene,pos,posf,view,qGIMob):
def checkCreate(scene,view,modelpath,string,num,event_pos,layoutPt):
    # The variable 'compt' will be empty when dropping cubeMesh,cyclMesh, but rest it shd be
    # compartment
    compt = view.sceneContainerPt.itemAt(view.mapToScene(event_pos))
    pos = view.mapToScene(event_pos)
    modelpath = moose.element(modelpath)
    if num:
        if string == "CubeMesh":
            string_num = "Compartment"+str(num)
        elif string == "CylMesh":
            string_num = "Compartment"+str(num)
        else:
            string_num = string+str(num)
    else:
        if string == "CubeMesh":
            string_num = "Compartment"
        elif string == "CylMesh":
            string_num = "Compartment"
        else:
            string_num = string
    if string == "CubeMesh":
        mobj = moose.CubeMesh(modelpath.path+'/'+string_num)
        mesh = moose.element(mobj.path+'/mesh')
        qGItem = ComptItem(scene,pos.toPoint().x(),pos.toPoint().y(),300,300,mobj)
        qGItem.setPen(QtGui.QPen(Qt.QColor(66,66,66,100), 5, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
        view.sceneContainerPt.addItem(qGItem)
        #layoutPt.setupSlot(mobj,qGItem)
        qGItem.cmptEmitter.connect(qGItem.cmptEmitter,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),layoutPt.positionChange1)
        qGItem.cmptEmitter.connect(qGItem.cmptEmitter,QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),layoutPt.objectEditSlot)
        compartment = qGItem
        layoutPt.qGraCompt[mobj]= qGItem
        #qGItem.prepareGeometryChange()
        view.emit(QtCore.SIGNAL("dropped"),mobj)
    elif string == "CylMesh":
        mobj = moose.CylMesh(modelpath.path+'/'+string_num)
        mesh = moose.element(mobj.path+'/mesh')
        qGItem = ComptItem(scene,pos.toPoint().x(),pos.toPoint().y(),300,300,mobj)
        qGItem.setPen(QtGui.QPen(Qt.QColor(66,66,66,100), 5, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
        view.sceneContainerPt.addItem(qGItem)
        layoutPt.setupSlot(mobj,qGItem)
        compartment = qGItem
        layoutPt.qGraCompt[mobj]= qGItem
        #qGItem.prepareGeometryChange()
        view.emit(QtCore.SIGNAL("dropped"),mobj)
        qGItem.cmptEmitter.connect(qGItem.cmptEmitter,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),layoutPt.positionChange1)
        qGItem.cmptEmitter.connect(qGItem.cmptEmitter,QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),layoutPt.objectEditSlot)
    elif string == "Pool" or string == "BufPool":
        #getting pos with respect to compartment otherwise if compartment is moved then pos would be wrong
        posWrtComp = (compt.mapFromScene(pos)).toPoint()
        mobj = compt.mobj
        print  "mobj ",mobj
        #parent = moose.element(mobj).parent
        if string == "Pool":
            poolObj = moose.Pool(mobj.path+'/'+string_num)

        else:
            poolObj = moose.BufPool(mobj.path+'/'+string_num)    
        qGItem =PoolItem(poolObj,compt)
        layoutPt.mooseId_GObj[poolObj.getId()] = qGItem
        posWrtComp = (compt.mapFromScene(pos)).toPoint()
        qGItem.setDisplayProperties(posWrtComp.x(),posWrtComp.y(),QtGui.QColor('green'),QtGui.QColor('blue'))
        layoutPt.setupSlot(poolObj,qGItem)
        view.emit(QtCore.SIGNAL("dropped"),poolObj)
        updateCompartmentSize(compt)

    elif  string == "Reac":
        posWrtComp = (compt.mapFromScene(pos)).toPoint()
        mobj = compt.mobj
        parent = moose.element(mobj).parent
        reacObj = moose.Reac(mobj.path+'/'+string_num)
        qGItem = ReacItem(reacObj,compt)
        reainfo = moose.element(reacObj).path+'/info'
        qGItem.setDisplayProperties(posWrtComp.x(),posWrtComp.y(),"white", "white")
        layoutPt.setupSlot(reacObj,qGItem)
        layoutPt.mooseId_GObj[reacObj.getId()] = qGItem
        view.emit(QtCore.SIGNAL("dropped"),reacObj)
        updateCompartmentSize(compt)
    elif  string == "StimulusTable":
        posWrtComp = (compt.mapFromScene(pos)).toPoint()
        mobj = compt.mobj
        parent = moose.element(mobj).parent
        tabObj = moose.StimulusTable(mobj.path+'/'+string_num)
        qGItem = TableItem(tabObj,compt)
        qGItem.setDisplayProperties(posWrtComp.x(),posWrtComp.y(),QtGui.QColor('white'),QtGui.QColor('white'))
        layoutPt.setupSlot(tabObj,qGItem)
        layoutPt.mooseId_GObj[tabObj.getId()] = qGItem
        view.emit(QtCore.SIGNAL("dropped"),tabObj)
        updateCompartmentSize(compt)
    elif string == "Function":
        posWrtComp = (compt.mapFromScene(pos)).toPoint()
        mobj = compt.mobj
        parent = moose.element(mobj).parent
        funObj = moose.Function(mobj.path+'/'+string_num)
        qGItem = FuncItem(funObj,compt)
        print "QGITem ",qGItem
        #qGItem.setPixmap(QtGui.QPixmap('../Docs/images/classIcon/Function.png'))
        print "qGItem",qGItem,isinstance(qGItem,KineticsDisplayItem)
        #GItem.setPos(posWrtComp.x(),posWrtComp.y())
        qGItem.setDisplayProperties(posWrtComp.x(),posWrtComp.y(),QtGui.QColor('red'),QtGui.QColor('green'))
        layoutPt.mooseId_GObj[funObj.getId()] = qGItem
        layoutPt.setupSlot(funObj,qGItem)
        view.emit(QtCore.SIGNAL("dropped"),funObj)
        updateCompartmentSize(compt)
    elif  string == "Enz" or string == "MMenz":
        #If 2 enz has same pool parent, then pos of the 2nd enz shd be displaced by some position, need to check how to deal with it
        posWrtComp = (compt.mapFromScene(pos)).toPoint()
        mobj = compt.mobj
        parent = moose.element(mobj).parent

        if string == "Enz":
            enzObj = moose.Enz(moose.element(mobj).path+'/'+string_num)
            moose.connect( enzObj, 'enz', mobj, 'reac' )
            enzPool = layoutPt.mooseId_GObj[mobj.getId()]
            poolboundingRect = enzPool.boundingRect()
            qGItem = EnzItem(enzObj,compt)
            posWrtPool = (enzPool.mapFromScene(pos)).toPoint()
            qGItem.setDisplayProperties(poolboundingRect.height()/2,poolboundingRect.height()-60,QtGui.QColor('green'),QtGui.QColor('blue'))
            layoutPt.setupSlot(enzObj,qGItem)
            layoutPt.mooseId_GObj[enzObj.getId()] = qGItem
            #Enz cplx need to be created
            Enz_cplx = enzObj.path+'/'+string_num+'_cplx';
            cplxItem = moose.Pool(Enz_cplx)
            qGEnz = layoutPt.mooseId_GObj[enzObj.getId()]
            qGItem = CplxItem(cplxItem,qGEnz)
            enzboundingRect = qGEnz.boundingRect()
            moose.connect( enzObj, 'cplx', cplxItem, 'reac' )
            qGItem.setDisplayProperties(enzboundingRect.height()/2,enzboundingRect.height()-60,QtGui.QColor('white'),QtGui.QColor('white'))
            layoutPt.setupSlot(cplxItem,qGItem)
            view.emit(QtCore.SIGNAL("dropped"),enzObj)

        else:
            enzObj = moose.MMenz(moose.element(mobj).path+'/'+string_num)
            #moose.connect( enzObj, 'enz', mobj, 'reac' )
            moose.connect(mobj,"nOut",enzObj,"enzDest")
            poolboundingRect = (layoutPt.mooseId_GObj[mobj.getId()]).boundingRect()
            qGItem = MMEnzItem(enzObj,compt)
            qGItem.setDisplayProperties(poolboundingRect.height()/2,poolboundingRect.height()-40,QtGui.QColor('green'),QtGui.QColor('blue'))
            layoutPt.setupSlot(enzObj,qGItem)
            layoutPt.mooseId_GObj[enzObj.getId()] = qGItem
            view.emit(QtCore.SIGNAL("dropped"),enzObj)
            #mobj = enzObj
        
        layoutPt.mooseId_GObj[enzObj.getId()] = qGItem
        compt = layoutPt.qGraCompt[moose.element(mobj).parent]
        updateCompartmentSize(compt)

def createObj(scene,view,modelpath,string,pos,layoutPt):
    event_pos = pos
    num = ''
    pos = view.mapToScene(event_pos)
    itemAt = view.sceneContainerPt.itemAt(pos)
    chemMesh = moose.wildcardFind(modelpath+'/##[ISA=ChemCompt]')
    if len(chemMesh) and (string == "CubeMesh" or string == "CylMesh"):
        QtGui.QMessageBox.information(None,'Drop Not possible','At present model building allowed only for  single compartment.',QtGui.QMessageBox.Ok)
        return
    if string == "Pool" or string == "BufPool" or string == "Reac" or string == "StimulusTable":
        if itemAt == None:
            QtGui.QMessageBox.information(None,'Drop Not possible','\'{newString}\' has to have compartment as its parent'.format(newString =string),QtGui.QMessageBox.Ok)
            return
    elif string == "Function":
        if itemAt == None:
            QtGui.QMessageBox.information(None,'Drop Not possible','\'{newString}\' has to have compartment as its parent'.format(newString =string),QtGui.QMessageBox.Ok)
            return

    elif string == "Enz" or string == "MMenz":
        if itemAt != None:
            if ((itemAt.mobj).className != "Pool" and (itemAt.mobj).className != "BufPool"):    
                QtGui.QMessageBox.information(None,'Drop Not possible','\'{newString}\' has to have Pool as its parent'.format(newString =string),QtGui.QMessageBox.Ok)
                return
        else:
            QtGui.QMessageBox.information(None,'Drop Not possible','\'{newString}\' has to have Pool as its parent'.format(newString =string),QtGui.QMessageBox.Ok)
            return

    if itemAt != None:
        if string in layoutPt.createdItem.keys():
            num = layoutPt.createdItem[string]
            layoutPt.createdItem[string]+=1
        else:
            layoutPt.createdItem[string] =1
    checkCreate(scene,view,modelpath,string,num,event_pos,layoutPt)
    #checkCreate(string,num,itemAt,qGraCompt,modelpath,scene,pos,posf,view,qGIMob)
    