from __future__ import absolute_import, print_function, division
# rmoogli.py: rdesigneur Moogli interface
# This is a fallback version designed to work with moogul but using almost
# the same interface as far as rdesigneur is concerned.
# Put in because the GL versions like moogli need all sorts of difficult 
# libraries and dependencies.
# Copyright (C) Upinder S. Bhalla NCBS 2018
# This program is licensed under the GNU Public License version 3.

import numpy as np
import rdesigneur.moogul as moogul
import moose
mooViews = []

class MooseNeuronDataWrapper( moogul.DataWrapper ):
    def __init__( self, neuronId, relativeObjPath, field ):
        moogul.DataWrapper.__init__( self, field )
        self.neuronId_ = neuronId
        self.relativeObjPath_ = relativeObjPath
        if field == "Ca":
            self.dummyObj = moose.CaConc( "/dummyCa" )
        elif field in ["Ik","Gk", "Ek", "Gbar", "modulation"]:
            self.dummyObj = moose.SynChan( "/dummyCa" )

        compts = moose.wildcardFind( neuronId.path + "/#[ISA=CompartmentBase]" )
        self.coords_ = np.array( [ii.coords for ii in compts] )
        self.getMinMax()
        if relativeObjPath == ".":
            self.objList_ = compts
        else:
            self.objList_ = []
            for i in compts:
                if moose.exists( i.path + '/' + relativeObjPath ):
                    elm = moose.element(i.path + '/' + relativeObjPath )
                    self.objList_.append( elm )
                else:
                    self.objList_.append( self.dummyObj )

    def getValues( self ):
        return np.array( [moose.getField(i, self.field_) for i in self.objList_] )
    
    def getCoords( self ):
        ''' Obtains 2-D array [comptidx, coord#] from the associated cell.
        There can be any number of rows, but only 7 columns (i.e, coords).
        These are x0, y0, z0, x, y, z, dia
        '''
        return self.coords_


class MooseChemDataWrapper( moogul.DataWrapper ):
    def __init__( self, objList, field ):
        moogul.DataWrapper.__init__( self, field )
        self.objList_ = objList
        if len( objList ) == 0:
            return
        #coords = np.array( [ii.coords for ii in objList] )
        self.coords_ = np.array( [ii.coords for ii in objList] )
        self.meshType_ = objList[0].compartment.className 
        if self.meshType_ in ["NeuroMesh", "CylMesh", "PsdMesh"]:
            # Unfortunately at present these return radius rather than
            # diameter in argument 6. To fix.
            self.coords_[:,6] *= 2.0
        elif self.meshType_ == "SpineMesh":
            # Spine entry has head[3], shaft[3], root[3], dia.
            # We want to put dia in idx == 6.
            self.coords_[:,6] = self.coords_[:,9]
        elif self.meshType_ == "PresynMesh":
            # This returns diameter in argument 6.
            # first vec is centre of base, second axis pointing 
            # toward postsyn
            # Hack: make each bouton as a cone with length == dia.
            diaColumn = np.array(self.coords_[:,6])
            diaColumn.shape = ( len( diaColumn ), 1 )
            self.coords_[:,3:6] = self.coords_[:,0:3] + self.coords_[:,3:6] * diaColumn
        elif self.meshType_ == "EndoMesh":
            # Returns centre as args 0,1,2, diameter as argument 3.
            # Make a sphere. The coords need to be set for 7 columns.
            temp = np.array( self.coords_ )
            self.coords_ = np.zeros( (temp.shape[0], 7 ) )
            self.coords_[:,0:3] = temp[:,0:3]
            self.coords_[:,3:6] = temp[:,0:3]
            self.coords_[:,6] = temp[:,3]
            #= np.array( self.coords_[0:3].extend( self.coords_[0:3] ).extend(self.coords_[3] ) )
        self.getMinMax()

    def getValues( self ):
        return np.array( [moose.getField(i, self.field_) for i in self.objList_] )
    
    def getCoords( self ):
        ''' Obtains 2-D array [comptidx, coord#] from the associated cell.
        There can be any number of rows, but only 7 columns (i.e, coords).
        These are x0, y0, z0, x, y, z, dia
        '''
        return self.coords_

    def meshType( self ):
        return self.meshType_


def makeMoogli( rd, mooObj, args, fieldInfo ):
    #mooObj is currently poorly handled. Ideally it should simply be a 
    # vector of objects to plot, each with coords. This could be readily
    # used to build up the display in a neutral manner.
    # Another version is to have a vector of objects to plot, only as a 
    # way to get at their fields. We would separately need mapping to
    # neuron compartments and index along length.
    # Cleaner still would be to have the C code give a vector of values
    # For now it means something different for chem and elec displays.
    #moogliEntry = [elecPath,bool,whichObjToDisplay,FieldToDisplay,titleForDisplay,rangeMin,rangeMax]
    mooField = args.field
    relObjPath = args.relpath
    numMoogli = len( mooObj )
    if args.ymin != args.ymax:
        ymin = args.ymin
        ymax = args.ymax
    else:
        ymin = fieldInfo[4]
        ymax = fieldInfo[5]
    #print( "fieldinfo = {}, ymin = {}, ymax = {}".format( fieldInfo, ymin, ymax ))

    viewer = moogul.MooView( title = args.title )
    if mooField == 'n' or mooField == 'conc':
        dw = MooseChemDataWrapper( mooObj, mooField )
        reacSystem = moogul.MooReacSystem( dw,
                valMin = ymin, valMax = ymax, diaScale = args.diaScale, fieldScale = fieldInfo[2] )
        viewer.addDrawable( reacSystem )
    else:
        dw = MooseNeuronDataWrapper( rd.elecid, relObjPath, mooField )
        neuron = moogul.MooNeuron( dw,
                valMin = ymin, valMax = ymax, diaScale = args.diaScale, fieldScale = fieldInfo[2] )
        viewer.addDrawable( neuron )
    return viewer

def updateMoogliViewer():
    simTime = moose.element( '/clock' ).currentTime
    for i in mooViews:
        i.updateValues( simTime )
    
def notifySimulationEnd():
    if len( mooViews ) > 0:
        mooViews[0].notifySimulationEnd()

def displayMoogli( rd, _dt, _runtime, rotation = 0.0, fullscreen = False, azim = 0.0, elev = 0.0, mergeDisplays = False, center = [0.0, 0.0, 0.0], colormap = 'jet', bg = 'default' ):
    global mooViews
    mooViews = rd.moogNames
    for view in rd.moogNames:
        view.firstDraw( mergeDisplays, rotation = rotation, azim = azim, elev = elev, center = center, colormap = colormap, bg = bg )
        # rotation in radians/frame, azim, elev in radians.
