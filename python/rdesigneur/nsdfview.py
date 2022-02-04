# nmoogli.py: nsdf Moogli interface
# This program displays NSDF files.
# Copyright (C) Upinder S. Bhalla NCBS 2022
# This program is licensed under the GNU Public License version 3.

import argparse
import numpy as np
import h5py
#import rdesigneur.moogul as moogul
import moogul
mooViews = []

class NsdfNeuronDataWrapper( moogul.DataWrapper ):
    def __init__( self, nsdf, neuronName, field ):
        moogul.DataWrapper.__init__( self, field )
        self.nsdf_ = nsdf
        self.neuronName_ = neuronName
        self.field_ = field
        self.simTime_ = 0.0
        self.idx_ = 0
        self.dt_= nsdf["/data/uniform/{}/{}".format( neuronName, field)].attrs['dt']
        self.coords_ = np.array( nsdf['/data/static/{}/coords'.format(neuronName) ] )
        self.getMinMax()

    def getValues( self ):
        ret=np.array( self.nsdf_["/data/uniform/ZombieCompartment/"+self.field_][self.idx_] )
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
        return nsdf["/data/uniform/ZombieCompartment/"+self.field_].shape

    def numObj( self ):
        return int( len( self.coords_ ) > 0 )

#####################################################################

class NsdfChemDataWrapper( moogul.DataWrapper ):
    def __init__( self, nsdf,  field ):
        moogul.DataWrapper.__init__( self, field )
        self.nsdf_ = nsdf
        self.field_ = field
        self.simTime_ = 0.0
        self.idx_ = 0
        self.dt_= nsdf["/data/uniform/{}/{}".format( "Pool", field)].attrs['dt']
        self.coords_ = np.array( nsdf['/data/static/{}/coords'.format("Pool") ] )
        print( self.coords_[0:10,] )
        print()
        print( self.coords_[300:310,] )
        print()
        print( self.coords_[600:610,] )
        if self.coords_.shape[1] == 10:
            self.coords_[:,6] = self.coords_[:,9] # Temp hack to get radius correctly
        print("COORDS SHAPE = ", self.coords_.shape )

        self.getMinMax()

    def getValues( self ):
        ret=np.array( self.nsdf_["/data/uniform/Pool/"+self.field_][self.idx_] )
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

    def meshType( self ):
        #return self.meshType_
        return "NeuroMesh"

    def getShape( self ):
        return self.nsdf_["/data/uniform/Pool/"+self.field_].shape

    def numObj( self ):
        return len( self.coords_ )

#####################################################################
def main():
    parser = argparse.ArgumentParser( description = "NSDF Moogli viewer." )
    parser.add_argument( "NSDF_filename", type = str, help = "Required: name of NSDF format File record of simulation" )
    parser.add_argument( "-v", "--viewspec", nargs = 4, default=[], help="Specification for each view: [objname, field, min, max]" )
    parser.add_argument( "-r", "--rotation", type = float, default=0.0, help="Rotate display around vertical axis by this angle in radians every step. Default 0.0")
    parser.add_argument( "-c", "--colormap", type = str, default="plasma", help="Name of matplotlib colormap to use. Default is 'plasma'")
    parser.add_argument( "-bg", "--background", type = str, default="default", help="Name of matplotlib color to use for background. Default is a light blue-gray.")
    parser.add_argument( '-m', '--merge_displays', action="store_true", help="Display all data in the same view." )
    parser.add_argument( '-l', '--list_datasets', action="store_true", help="List possible datasets available to view." )
    args = parser.parse_args()

    nsdf = h5py.File( args.NSDF_filename, 'r' )

    viewer = []
    for idx in range( 0, len( args.viewspec ), 4):
        viewer.append( makeMoogli( nsdf, args.viewspec[idx:idx+4] ) )

    if len( args.viewspec ) == 0:
        # Check if the nsdf file has a ZombieCompt or a Pool entry and 
        # try to display it.
        print( "warning: No viewpsec defined in command line" )
        #viewer.append( makeMoogli( nsdf, args.viewspec[idx:idx+4] ) )
        quit()

    shape = viewer[0].drawables_[0].dataWrapper_.getShape()
    numSteps = shape[1]
    dt = viewer[0].drawables_[0].dataWrapper_.dt_

    for idx in range( 0, len( args.viewspec ), 4):
        viewer[idx].firstDraw( args.merge_displays, rotation = args.rotation, colormap = args.colormap, bg = args.background )

    for step in range( numSteps ):
        simTime = 0.0
        for idx in range( 0, len( args.viewspec ), 4):
            viewer[idx].updateValues( simTime )
            simTime += dt

    while True:
        sleep(1)


def makeMoogli( nsdf, args ):
    objname, field, symin, symax = args
    ymin = float( symin )
    ymax = float( symax )
    fieldScale = 1.0
    diaScale = 1.0

    viewer = moogul.MooView( title = objname + "." + field )
    if field == 'n' or field == 'conc':
        dw = NsdfChemDataWrapper( nsdf, field )
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
