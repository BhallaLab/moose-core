import moose
from kkitQGraphics import * 
from kkitOrdinateUtil import *
from kkitUtil import *
import PyQt4
from setsolver import *

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
    #view.fitInView(view.sceneContainerPt.itemsBoundingRect())
# def checkCreate(string,num,itemAt,qGraCompt,modelRoot,scene,pos,posf,view,qGIMob):
def checkCreate(scene,view,modelpath,string,num,event_pos,layoutPt):
    # The variable 'compt' will be empty when dropping cubeMesh,cyclMesh, but rest it shd be
    # compartment
    if modelpath.find('/',1) > -1:
        modelRoot = modelpath[0:modelpath.find('/',1)]
    else:
        modelRoot = modelpath
    mType = moose.Annotator((moose.element(modelRoot+'/info'))).modeltype
    itemAtView = view.sceneContainerPt.itemAt(view.mapToScene(event_pos))
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

    if string == "Pool" or string == "BufPool" or string == "Reac" or string == "StimulusTable":
        compartment = None
        itemClass = type(itemAtView).__name__
        if ( itemClass == 'QGraphicsRectItem'):
            mobj = itemAtView.parentItem().mobj
        else:
            mobj = itemAtView.mobj
        compartment = findCompartment(mobj)
        mobj = compartment
    if string == "CubeMesh":
        mobj = moose.CubeMesh(modelpath.path+'/'+string_num)
        mobj.volume = 1e-15
        mesh = moose.element(mobj.path+'/mesh')
        print "modelBuid ",view.sceneContainerPt.sceneRect()
        qGItem = ComptItem(scene,pos.toPoint().x(),pos.toPoint().y(),800,500,mobj)
        qGItem.setPen(QtGui.QPen(Qt.QColor(66,66,66,100), 5, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
        view.sceneContainerPt.addItem(qGItem)
        qGItem.cmptEmitter.connect(qGItem.cmptEmitter,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),layoutPt.positionChange1)
        qGItem.cmptEmitter.connect(qGItem.cmptEmitter,QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),layoutPt.objectEditSlot)
        compartment = qGItem
        layoutPt.qGraCompt[mobj]= qGItem
        view.emit(QtCore.SIGNAL("dropped"),mobj)
    elif string == "CylMesh":
        mobj = moose.CylMesh(modelpath.path+'/'+string_num)
        mobj.volume = 1e-15
        mesh = moose.element(mobj.path+'/mesh')
        qGItem = ComptItem(scene,pos.toPoint().x(),pos.toPoint().y(),800,500,mobj)
        qGItem.setPen(QtGui.QPen(Qt.QColor(66,66,66,100), 5, Qt.Qt.SolidLine, Qt.Qt.RoundCap, Qt.Qt.RoundJoin))
        view.sceneContainerPt.addItem(qGItem)
        compartment = qGItem
        layoutPt.qGraCompt[mobj]= qGItem
        view.emit(QtCore.SIGNAL("dropped"),mobj)
        qGItem.cmptEmitter.connect(qGItem.cmptEmitter,QtCore.SIGNAL("qgtextPositionChange(PyQt_PyObject)"),layoutPt.positionChange1)
        qGItem.cmptEmitter.connect(qGItem.cmptEmitter,QtCore.SIGNAL("qgtextItemSelectedChange(PyQt_PyObject)"),layoutPt.objectEditSlot)

    elif string == "Pool" or string == "BufPool":
        #getting pos with respect to compartment otherwise if compartment is moved then pos would be wrong
        posWrtComp = (itemAtView.mapFromScene(pos)).toPoint()
        # mobj = itemAtView.mobj
        # compartment = None
        # compartment = findCompartment(mobj)
        # if not isinstance(compartment,ChemCompt):
        #     print " Dropping not possible, Pool needs Compartment as its parent"
        # else:
            # mobj = compartment
        if string == "Pool":
            poolObj = moose.Pool(mobj.path+'/'+string_num)
            poolinfo = moose.Annotator(poolObj.path+'/info')
        else:
            poolObj = moose.BufPool(mobj.path+'/'+string_num)    
            poolinfo = moose.Annotator(poolObj.path+'/info')
        qGItem =PoolItem(poolObj,itemAtView)
        layoutPt.mooseId_GObj[poolObj] = qGItem
        posWrtComp = (itemAtView.mapFromScene(pos)).toPoint()
        bgcolor = getRandColor()
        qGItem.setDisplayProperties(posWrtComp.x(),posWrtComp.y(),QtGui.QColor('green'),bgcolor)
        poolinfo.color = str(bgcolor.getRgb())
        if mType == "new_kkit":
            poolinfo.x = posWrtComp.x()
            poolinfo.y = posWrtComp.y()
        view.emit(QtCore.SIGNAL("dropped"),poolObj)
        setupItem(modelpath.path,layoutPt.srcdesConnection)
        layoutPt.drawLine_arrow(False)
        #Dropping is on compartment then update Compart size
        if isinstance(mobj,moose.ChemCompt):
            compt = layoutPt.qGraCompt[moose.element(mobj)]
            updateCompartmentSize(compt)
        
    elif  string == "Reac":
        posWrtComp = (itemAtView.mapFromScene(pos)).toPoint()
        reacObj = moose.Reac(mobj.path+'/'+string_num)
        reacinfo = moose.Annotator(reacObj.path+'/info')
        qGItem = ReacItem(reacObj,itemAtView)
        qGItem.setDisplayProperties(posWrtComp.x(),posWrtComp.y(),"white", "white")
        if mType == "new_kkit":
            reacinfo.x = posWrtComp.x()
            reacinfo.y = posWrtComp.y()
        layoutPt.mooseId_GObj[reacObj] = qGItem
        view.emit(QtCore.SIGNAL("dropped"),reacObj)
        setupItem(modelpath.path,layoutPt.srcdesConnection)
        layoutPt.drawLine_arrow(False)
        #Dropping is on compartment then update Compart size
        if isinstance(mobj,moose.ChemCompt):
            compt = layoutPt.qGraCompt[moose.element(mobj)]
            updateCompartmentSize(compt)

    elif  string == "StimulusTable":
        posWrtComp = (itemAtView.mapFromScene(pos)).toPoint()
        tabObj = moose.StimulusTable(mobj.path+'/'+string_num)
        tabinfo = moose.Annotator(tabObj.path+'/info')
        qGItem = TableItem(tabObj,itemAtView)
        qGItem.setDisplayProperties(posWrtComp.x(),posWrtComp.y(),QtGui.QColor('white'),QtGui.QColor('white'))
        if mType == "new_kkit":
            tabinfo.x = posWrtComp.x()
            tabinfo.y = posWrtComp.y()
        layoutPt.mooseId_GObj[tabObj] = qGItem
        view.emit(QtCore.SIGNAL("dropped"),tabObj)
        setupItem(modelpath.path,layoutPt.srcdesConnection)
        layoutPt.drawLine_arrow(False)
        #Dropping is on compartment then update Compart size
        if isinstance(mobj,moose.ChemCompt):
            compt = layoutPt.qGraCompt[moose.element(mobj)]
            updateCompartmentSize(compt)
    elif string == "Function":
        posWrtComp = (itemAtView.mapFromScene(pos)).toPoint()
        mobj = itemAtView.mobj
        funcObj = moose.Function(mobj.path+'/'+string_num)
        funcObj.numVars+=1
        funcinfo = moose.Annotator(funcObj.path+'/info')
        moose.connect( funcObj, 'valueOut', mobj.path ,'setN' )
        funcParent = layoutPt.mooseId_GObj[element(mobj.path)]
        qGItem = FuncItem(funcObj,funcParent)
        qGItem.setDisplayProperties(posWrtComp.x(),posWrtComp.y(),QtGui.QColor('red'),QtGui.QColor('green'))
        layoutPt.mooseId_GObj[funcObj] = qGItem
        if mType == "new_kkit":
            funcinfo.x = posWrtComp.x()
            funcinfo.y = posWrtComp.y()
        view.emit(QtCore.SIGNAL("dropped"),funcObj)
        setupItem(modelpath.path,layoutPt.srcdesConnection)
        layoutPt.drawLine_arrow(False)
        #Dropping is on compartment then update Compart size
        mooseCmpt = findCompartment(mobj)
        if isinstance(mooseCmpt,moose.ChemCompt):
            compt = layoutPt.qGraCompt[moose.element(mooseCmpt)]
            updateCompartmentSize(compt)

    elif  string == "Enz" or string == "MMenz":
        #If 2 enz has same pool parent, then pos of the 2nd enz shd be displaced by some position, need to check how to deal with it
        #posWrtComp = (itemAtView.mapFromScene(pos)).toPoint()
        posWrtComp = pos
        mobj = itemAtView.mobj
        enzPool = layoutPt.mooseId_GObj[mobj]
        if ((mobj.parent).className == "Enz"):
            QtGui.QMessageBox.information(None,'Drop Not possible','\'{newString}\' has to have Pool as its parent and not Enzyme Complex'.format(newString =string),QtGui.QMessageBox.Ok)
            return
        else:
            enzparent = findCompartment(mobj)
            parentcompt = layoutPt.qGraCompt[enzparent]
        if string == "Enz":
            enzObj = moose.Enz(moose.element(mobj).path+'/'+string_num)
            enzinfo = moose.Annotator(enzObj.path+'/info')
            moose.connect( enzObj, 'enz', mobj, 'reac' )
            qGItem = EnzItem(enzObj,parentcompt)
            layoutPt.mooseId_GObj[enzObj] = qGItem
            posWrtComp = pos
            bgcolor = getRandColor()
            qGItem.setDisplayProperties(posWrtComp.x(),posWrtComp.y()-40,QtGui.QColor('green'),bgcolor)
            if mType == "new_kkit":
                enzinfo.x = posWrtComp.x()
                enzinfo.y = posWrtComp.y()
            enzinfo.color = str(bgcolor.name())
            e = moose.Annotator(enzinfo)
            e.x = posWrtComp.x()
            e.y = posWrtComp.y()
            Enz_cplx = enzObj.path+'/'+string_num+'_cplx';
            cplxItem = moose.Pool(Enz_cplx)
            cplxinfo = moose.Annotator(cplxItem.path+'/info')
            qGEnz = layoutPt.mooseId_GObj[enzObj]
            qGItem = CplxItem(cplxItem,qGEnz)
            layoutPt.mooseId_GObj[cplxItem] = qGItem
            enzboundingRect = qGEnz.boundingRect()
            moose.connect( enzObj, 'cplx', cplxItem, 'reac' )
            qGItem.setDisplayProperties(enzboundingRect.height()/2,enzboundingRect.height()-40,QtGui.QColor('white'),QtGui.QColor('white'))
            cplxinfo.x = enzboundingRect.height()/2
            cplxinfo.y = enzboundingRect.height()-60
            view.emit(QtCore.SIGNAL("dropped"),enzObj)

        else:
            enzObj = moose.MMenz(mobj.path+'/'+string_num)
            enzinfo = moose.Annotator(enzObj.path+'/info')
            moose.connect(mobj,"nOut",enzObj,"enzDest")
            qGItem = MMEnzItem(enzObj,parentcompt)
            posWrtComp = pos
            bgcolor = getRandColor()
            qGItem.setDisplayProperties(posWrtComp.x(),posWrtComp.y()-30,QtGui.QColor('green'),bgcolor)
            enzinfo.x = posWrtComp.x()
            enzinfo.y = posWrtComp.y()
            enzinfo.color = str(bgcolor.name())
            layoutPt.mooseId_GObj[enzObj] = qGItem
            view.emit(QtCore.SIGNAL("dropped"),enzObj)
        setupItem(modelpath.path,layoutPt.srcdesConnection)
        layoutPt.drawLine_arrow(False)
        #Dropping is on compartment then update Compart size
        if isinstance(enzparent,moose.ChemCompt):
            updateCompartmentSize(parentcompt)

def createObj(scene,view,modelpath,string,pos,layoutPt):
    event_pos = pos
    num = ''
    pos = view.mapToScene(event_pos)
    itemAt = view.sceneContainerPt.itemAt(pos)
    chemMesh = moose.wildcardFind(modelpath+'/##[ISA=ChemCompt]')
    deleteSolver(modelpath)
    if len(chemMesh) and (string == "CubeMesh" or string == "CylMesh"):
        QtGui.QMessageBox.information(None,'Drop Not possible','At present model building allowed only for  single compartment.',QtGui.QMessageBox.Ok)
        return
    if string == "Pool" or string == "BufPool" or string == "Reac" or string == "StimulusTable":
        if itemAt == None:
            QtGui.QMessageBox.information(None,'Drop Not possible','\'{newString}\' has to have compartment as its parent'.format(newString =string),QtGui.QMessageBox.Ok)
            return
    elif string == "Function":
        if itemAt != None:
            if ((itemAt.mobj).className != "BufPool"):    
                QtGui.QMessageBox.information(None,'Drop Not possible','\'{newString}\' has to have BufPool as its parent'.format(newString =string),QtGui.QMessageBox.Ok)
                return
        else:
            QtGui.QMessageBox.information(None,'Drop Not possible','\'{newString}\' has to have BufPool as its parent'.format(newString =string),QtGui.QMessageBox.Ok)
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
    
def findCompartment(mooseObj):
    if mooseObj.path == '/':
        return None
    elif isinstance(mooseObj,ChemCompt):
        return (mooseObj)
    else:
        return findCompartment(moose.element(mooseObj.parent))