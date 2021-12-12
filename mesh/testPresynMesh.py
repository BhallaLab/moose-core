import moose
import rdesigneur as rd

rdes = rd.rdesigneur(
    elecDt = 50e-6,
    chemDt = 0.002,
    # cellProto syntax: ['ballAndStick', 'name', somaDia, somaLength, dendDia, dendLength, numDendSegments ]
    cellProto = [['ballAndStick', 'soma', 12e-6, 12e-6, 4e-6, 100e-6, 2 ]],
    spineProto = [['makeActiveSpine()', 'spine']],
    spineDistrib = [['spine', '#dend#', '50e-6', '1e-6']],
)

moose.seed(123)
rdes.buildModel()
moose.reinit()

pre1 = moose.PresynMesh( 'pre1' )
spineHeads = moose.wildcardFind( "/model/elec/#head#" )
print( spineHeads )
pre1.buildOnSpineHeads( spineHeads )

moose.showfield( pre1 )
print( "pre1 voxelVolume = ", pre1.voxelVolume )
print( "pre1 voxelMidpoint = ", pre1.voxelMidpoint )


pre2 = moose.PresynMesh( 'pre2' )
dends = moose.wildcardFind( "/model/elec/#dend#" )
print( dends )
pre2.buildOnDendrites( dends , 20e-6 )

moose.showfield( pre2 )
print( "pre2 voxelVolume = ", pre2.voxelVolume )
print( "pre2 voxelMidpoint = ", pre2.voxelMidpoint )


