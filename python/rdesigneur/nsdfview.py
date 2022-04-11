# nmoogli.py: nsdf Moogli interface
# This program displays NSDF files.
# Copyright (C) Upinder S. Bhalla NCBS 2022
# This program is licensed under the GNU Public License version 3.

import argparse
import numpy as np
import h5py
import time
#import rdesigneur.moogul as moogul
import moogul
mooViews = []

defaultFieldRange = { 'conc': [0.0, 1.0], 'n': [0, 10], 'Vm': [-0.7, 0.02], 'Ik':[-1e-9, 1e-9], 'Ca':[0.0, 1.0] }

class ObjHandle:
    def __init__( self, path ):
        self.path = str( path, "utf-8" )

class NsdfNeuronDataWrapper( moogul.DataWrapper ):
    def __init__( self, nsdf, path, field ):
        moogul.DataWrapper.__init__( self, field )
        self.nsdf_ = nsdf
        self.path_ = path
        spl = path.split( '/', 1 )
        self.objBase_ = "%elec"
        if len( spl ) > 1:  # Some relative path tree.
            if spl[0] == "#":
                self.objRel_ = "##[ISA=CompartmentBase]%" + spl[1].replace( '/', '%' )
            else:
                self.objRel_ = spl[0] + "%" + spl[1].replace( '/', '%' )
        else:
            if spl[0] == "#":
                self.objRel_ = "##[ISA=CompartmentBase]"
            else:
                self.objRel_ = path   # No slashes in the string.
        self.field_ = field
        self.simTime_ = 0.0
        self.idx_ = 0
        self.dt_= nsdf["/data/uniform/{}/{}/{}".format( self.objBase_, self.objRel_, field)].attrs['dt']
        self.coords_ = np.array( nsdf['/data/static/{}/{}/coords'.format(self.objBase_, self.objRel_) ] )
        objPaths = nsdf["/map/static/{}/{}/coords".format( self.objBase_, self.objRel_)]
        self.objList_ = [ ObjHandle( path ) for path in objPaths ]
        #print( "COORDS SHAPE for ", neuronName, " = ", self.coords_.shape )
        self.getMinMax()

    def getValues( self ):
        npath = "/data/uniform/{}/{}/{}".format( self.objBase_, self.objRel_, self.field_ )
        ret=np.array( self.nsdf_[npath][:,self.idx_] )
        #ret = np.array( nsdf["/data/uniform/{}/{}".format( neuronName, field)][self.idx_] )
        self.idx_ += 1
        self.simTime_ = self.idx_ * self.dt_
        return ret

    def getCoords( self ):
        ''' Obtains 2-D array [comptidx, coord#] from the associated cell.
        There can be any number of rows, but only 7 columns (i.e, coords).
        These are x0, y0, z0, x, y, z, dia
        '''
        return self.coords_

    def getShape( self ):
        npath = "/data/uniform/{}/{}/{}".format( self.objBase_, self.objRel_, self.field_ )
        return self.nsdf_[npath].shape

    def numObj( self ):
        return len( self.coords_ )

    def getHistory( self, path, field ):
        if field != self.field_:
            print( "NsdfNeuronDataWrapper:getHistory Error: field name does not match: ", field, self.field_ )
            return
        objIdx = [ idx for idx, val in enumerate( self.objList_ ) if val.path == path ]
        if len( objIdx ) == 0:
            print( "NsdfNeuronDataWrapper:getHistory Error: path not found: ",  path )
            return

        npath = "/data/uniform/{}/{}/{}".format( self.objBase_, self.objRel_, self.field_ )
        val = self.nsdf_[npath][objIdx[0],:]
        t = np.arange( 0.0, len( val ), 1.0 ) * self.dt_
        return t, val

#####################################################################

class NsdfChemDataWrapper( moogul.DataWrapper ):
    def __init__( self, nsdf, objname,  field ):
        if len( objname.split('/' ) ) != 2:
            print( "Error: objname needs '/' between base and rel parts: ", objname )
            assert( 0 )
        moogul.DataWrapper.__init__( self, field )
        self.nsdf_ = nsdf
        spl = objname.split( '/', 1 )
        if len( spl ) > 1:  # Can't deal with case where we have /chem as base but multiple levels below it.
            self.objBase_ = '%' + spl[0]
            self.objRel_ = spl[1].replace( '/', '%' )
            if self.objRel_[-1] != ']':  # put in wildcard, if none provided
                self.objRel_ += '[]'
        else:   # obj is direct child of /chem.
            self.objBase_ = "%chem"
            self.objRel_ = objname  # doesn't have any slashes either.
            if self.objRel_[-1] != ']':  # put in wildcard, if none provided
                self.objRel_ += '[]'

        self.field_ = field
        self.simTime_ = 0.0
        self.idx_ = 0
        self.dt_= nsdf["/data/uniform/{}/{}/{}".format( self.objBase_, self.objRel_, field)].attrs['dt']
        self.coords_ = np.array( nsdf['/data/static/{}/{}/coords'.format(self.objBase_, self.objRel_) ] )
        if self.coords_.shape[1] == 10:
            self.coords_[:,6] = self.coords_[:,9] # Temp hack to get radius correctly
        objPaths = nsdf["/map/static/{}/{}/coords".format( self.objBase_, self.objRel_)]
        self.objList_ = [ ObjHandle( path ) for path in objPaths ]

        self.getMinMax()

    def getValues( self ):
        npath = "/data/uniform/{}/{}/{}".format( self.objBase_, self.objRel_, self.field_ )
        ret=np.array( self.nsdf_[npath][:,self.idx_] )
        self.idx_ += 1
        self.simTime_ = self.idx_ * self.dt_
        return ret

    def getCoords( self ):
        ''' Obtains 2-D array [comptidx, coord#] from the associated cell.
        There can be any number of rows, but only 7 columns (i.e, coords).
        These are x0, y0, z0, x, y, z, dia
        '''
        return self.coords_

    def meshType( self ):
        #return self.meshType_
        return "NeuroMesh"

    def getShape( self ):
        npath = "/data/uniform/{}/{}/{}".format( self.objBase_, self.objRel_, self.field_ )
        return self.nsdf_[npath].shape

    def numObj( self ):
        return len( self.coords_ )

    def getHistory( self, path, field ):
        if field != self.field_:
            print( "NsdfChemDataWrapper:getHistory Error: field name does not match: ", field, self.field_ )
            return
        objIdx = [ idx for idx, val in enumerate( self.objList_ ) if val.path == path ]
        if len( objIdx ) == 0:
            print( "NsdfChemDataWrapper:getHistory Error: path not found: ",  path )
            return

        npath = "/data/uniform/{}/{}/{}".format( self.objBase_, self.objRel_, self.field_ )
        val = self.nsdf_[npath][objIdx[0],:]
        t = np.arange( 0, len( val ), 1 ) * self.dt_
        return t, val

#####################################################################
def printElecDatasetInfo( nsdf ):
    elec = nsdf["/data/uniform/%elec"]
    for ee in elec:
        path = ""
        spl = ee.split( '%', 1 )
        if spl[0] == "##[ISA=CompartmentBase]":
            path += "#"
        else:
            path = spl[0]
        if len( spl ) > 1:
            path += '/' + spl[1].replace( '%', '/' )
        for datasetName in elec[ee]:
            path += '.' + datasetName
            dataset = elec[ee + '/' + datasetName]
            dt = dataset.attrs['dt']
            shape = dataset.shape
            print( "Elec: {:<36} shape=({}),  dt={}".format( path, shape, dt ) )

def printDatasetInfo( nsdf ):
    uniform = nsdf["/data/uniform"]
    for uu in uniform:
        if uu == "%elec":
            printElecDatasetInfo( nsdf )
        else:
            group = uniform[uu]
            path = uu[1:]
            for ch in group:
                path += '/' + ch.replace( '%', '/' )
                for datasetName in group[ch]:
                    dataset = group[ ch + '/' + datasetName]
                    #print( "ATTR NAMES = ", dataset.keys() )
                    dt = dataset.attrs['dt']
                    shape = dataset.shape
                    print( "Chem: {:<36} shape=({}),  dt={}".format( path + '.' + datasetName, shape, dt ) )

def listDatasets( nsdf ):
    ## Returns list of datasets in path.field format.
    ret = []
    uniform = nsdf["/data/uniform"]
    for uu in uniform:
        if uu == "%elec":
            elec = nsdf["/data/uniform/%elec"]
            for ee in elec:
                path = ""
                spl = ee.split( '%', 1 )
                if spl[0] == "##[ISA=CompartmentBase]":
                    path += "#"
                else:
                    path = spl[0]
                if len( spl ) > 1:
                    path += '/' + spl[1].replace( '%', '/' )
                for field in elec[ee]:
                    ret.append( path + '.' + field )
        else:
            group = uniform[uu]
            for ch in group:
                path = uu[1:] + '/' + ch.replace( '%', '/' )
                if path[-2:] == '[]':
                    path = path[:-2]
                for field in group[ch]:
                    ret.append( path + '.' + field )
    return ret

#####################################################################
def main():
    parser = argparse.ArgumentParser( description = "NSDF Moogli viewer." )
    parser.add_argument( "NSDF_filename", type = str, help = "Required: name of NSDF format File record of simulation" )
    parser.add_argument( "-v", "--viewspec", nargs = '+', default=[], action='append',  help="Specification for each view: [objname, field, min, max]. Any number of views may be specified, each indicated by -v or --viewspec." )
    parser.add_argument( "-r", "--rotation", type = float, default=0.0, help="Rotate display around vertical axis by this angle in radians every step. Default 0.0")
    parser.add_argument( "-c", "--colormap", type = str, default="plasma", help="Name of matplotlib colormap to use. Default is 'plasma'")
    parser.add_argument( "-bg", "--background", type = str, default="default", help="Name of matplotlib color to use for background. Default is a light blue-gray.")
    parser.add_argument( '-m', '--merge_displays', action="store_true", help="Display all data in the same view." )
    parser.add_argument( '-l', '--list_datasets', action="store_true", help="List possible datasets available to view." )
    args = parser.parse_args()

    '''
    if (not args.list_datasets) and len( args.viewspec ) == 0:
        print( "warning: No viewpsec defined in command line" )
        quit()
    '''

    nsdf = h5py.File( args.NSDF_filename, 'r' )
    if args.list_datasets:
        printDatasetInfo( nsdf )
        quit()

    if len( args.viewspec ) == 0:
        viewspec = []
        datasets = listDatasets( nsdf )
        for ds in datasets:
            spl = ds.split( '.' )
            dispMin, dispMax = defaultFieldRange[ spl[1] ]
            viewspec.append( [spl[0], spl[1], dispMin, dispMax ] )
        merge_displays = True
    else:
        viewspec = args.viewspec
        merge_displays = args.merge_displays

    viewer = []
    for vs in viewspec:
        viewer.append( makeMoogli( nsdf, vs ) )
    dt = viewer[0].drawables_[0].dataWrapper_.dt_
    shape = viewer[0].drawables_[0].dataWrapper_.getShape()
    numSteps = shape[1]
    for v in viewer:
        v.firstDraw( merge_displays, rotation = args.rotation, colormap = args.colormap, bg = args.background )

    simTime = 0.0
    for step in range( numSteps ):
        for v in viewer:
            v.updateValues( simTime )
        simTime += dt
    viewer[0].notifySimulationEnd()

    while True:
        time.sleep(1)


def makeMoogli( nsdf, args ):
    objname, field, symin, symax = args
    ymin = float( symin )
    ymax = float( symax )
    fieldScale = 1.0
    diaScale = 1.0

    viewer = moogul.MooView( title = objname + "." + field )
    if field == 'n' or field == 'conc':
        dw = NsdfChemDataWrapper( nsdf, objname, field )
        if field == 'conc':
            fieldScale = 1.0e3
        reacSystem = moogul.MooReacSystem( dw,
                valMin = ymin, valMax = ymax, diaScale = diaScale, fieldScale = fieldScale )
        viewer.addDrawable( reacSystem )
    else:
        dw = NsdfNeuronDataWrapper( nsdf, objname, field )
        neuron = moogul.MooNeuron( dw,
                valMin = ymin, valMax = ymax, diaScale = diaScale, fieldScale = fieldScale )
        viewer.addDrawable( neuron )
    return viewer

if __name__ == "__main__":
    main()
