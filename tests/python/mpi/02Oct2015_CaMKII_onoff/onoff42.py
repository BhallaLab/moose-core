########################################################################
# This program is copyright (c) Upinder S. Bhalla, NCBS, 2015.
# It is licenced under the GPL 2.1 or higher.
# There is no warranty of any kind. You are welcome to make copies under 
# the provisions of the GPL.
# This programme illustrates building a panel of multiscale models to
# test neuronal plasticity in different contexts.
########################################################################

import numpy
import time
import pylab
import moose
from moose import neuroml
import matplotlib.pyplot as plt
import sys
import os
from moose.neuroml.ChannelML import ChannelML
import rdesigneur as rd

PI = 3.14159265359
useGssa = True
combineSegments = False
baselineTime = 10
tetTime = 1
interTetTime = 20
postTetTime = 24
ltdTime = 36.0
postLtdTime = 60
do3D = False
dt = 0.001
plotdt = 0.1
psdTetCa = 8e-3
basalCa = 0.08e-3
ltdCa = 0.25e-3

def buildRdesigneur():
    ##################################################################
    # Here we define which prototypes are to be loaded in to the system.
    # Each specification has the format
    # source [localName]
    # source can be any of
    # filename.extension,   # Identify type of file by extension, load it.
    # function(),           # func( name ) builds object of specified name
    # file.py:function() ,  # load Python file, run function(name) in it.
    # moose.Classname       # Make obj moose.Classname, assign to name.
    # path                  # Already loaded into library or on path.
    # After loading the prototypes, there should be an object called 'name'
    # in the library.
    ##################################################################
    cellProto = [ ['../cells/ca1_minimal.p', 'elec'] ]
    spineProto = [ ['makeSpineProto()', 'spine' ]]
    chemProto = [ ['../chem/CaMKII_merged42.g', 'chem'] ]

    ##################################################################
    # Here we define what goes where, and any parameters. Each distribution
    # has the format
    # protoName, path, field, expr, [field, expr]...
    # where 
    #   protoName identifies the prototype to be placed on the cell
    #   path is a MOOSE wildcard path specifying where to put things
    #   field is the field to assign.
    #   expr is a math expression to define field value. This uses the
    #     muParser. Built-in variables are p, g, L, len, dia.
    #     The muParser provides most math functions, and the Heaviside 
    #     function H(x) = 1 for x > 0 is also provided.
    ##################################################################
    chemRange = "H(1.1e-6 - dia) * H(p - 1300e-6)"
    spineDistrib = [ \
            ["spine", '#apical#', \
                "spineSpacing", chemRange + " * 5e-6", \
                "spineSpacingDistrib", "1e-6", \
                "angle", "0", \
                "angleDistrib", "0", \
                "size", "1", \
                "sizeDistrib", "0" ] \
        ]
    chemDistrib = [ \
            [ "chem", "#apical#", "install", chemRange ] 
        ]

    ######################################################################
    # Here we define the mappings across scales. Format:
    # sourceObj sourceField destObj destField couplingExpr [wildcard][spatialExpn]
    # where the coupling expression is anything a muParser can evaluate,
    # using the input variable x. For example: 8e-5 + 300*x
    # For now, let's use existing adaptors which take an offset and scale.
    ######################################################################
    adaptorList = [
        [ 'Ca_conc', 'Ca', 'psd/Ca_input', 'concInit', 8e-5, 1 ],
        [ 'Ca_conc', 'Ca', 'dend/Ca_dend_input', 'concInit', 8e-5, 1 ],
        [ 'psd/tot_PSD_R', 'n', 'glu', 'modulation', 0.1, 0.01 ],
    ]

    ######################################################################
    # Having defined everything, now to create the rdesigneur and proceed
    # with creating the model.
    ######################################################################
    
    rdes = rd.rdesigneur(
        useGssa = useGssa, \
        combineSegments = combineSegments, \
        stealCellFromLibrary = True, \
        spineDistrib = spineDistrib, \
        chemDistrib = chemDistrib, \
        cellProto = cellProto, \
        spineProto = spineProto, \
        chemProto = chemProto
    )

    return rdes

def buildOnePlot( path ):
    elist = moose.vec( '/model/chem/' + path )
    assert( len( elist ) > 0 )
    tabname = path.replace( '/', '_' )
    tab = moose.Table2( '/graphs/' + tabname, len( elist ) ).vec
    moose.connect( tab, 'requestOut', elist, 'getConc', 'OneToOne' )

def buildPlots():
    if not moose.exists( '/graphs' ):
        moose.Neutral( '/graphs' )
    for i in ( 'psd/Ca', 'spine/Ca', 'dend/DEND/Ca', \
            'psd/CaM_dash_Ca4', 'spine/CaM_dash_Ca4', \
            'dend/DEND/CaM/CaM_dash_Ca4', \
            'psd/CaM', 'spine/CaM', \
            'dend/DEND/CaM/CaM', \
            'psd/actCaMKII', 'spine/actCaMKII', 'dend/DEND/actCaMKII', \
            'psd/CaMKII', 'spine/CaMKII', \
            'dend/DEND/CaMKII_BULK/CaMKII', \
            'psd/tot_dash_CaM_dash_CaMKII', 'spine/tot_dash_CaM_dash_CaMKII', \
            'dend/DEND/CaMKII_BULK/tot_dash_CaM_dash_CaMKII', \
            'psd/PP1_dash_active', 'spine/CaMKII_dash_thr286', \
            'dend/DEND/CaMKII_BULK/CaMKII_dash_thr286', \
            'psd/CaM_CaN', 'spine/CaM_CaN',  \
            'psd/tot_PSD_R' \
            ):
        print i
        buildOnePlot( i )

def displayPlots():
    fig = plt.figure( figsize=(20,12 ) )
    elist = moose.wildcardFind( '/graphs/#[0]' )
    n = len(elist)
    j = 1
    timePts = numpy.arange( 0, len( elist[0].vector ) ) * elist[0].dt
    for i in elist:
        plt.subplot( 1 + n/3, 3, j )
        j = j + 1
        for k in range( len(i.vec) ):
            plt.plot( timePts, i.vec[k].vector )
        plt.title( i.name )
        print i.name
    plt.show()

def main():
    numpy.random.seed( 1234 )
    rdes = buildRdesigneur()
    rdes.buildModel( '/model' )
    assert( moose.exists( '/model' ) )
    moose.element( '/model/elec/hsolve' ).tick = -1
    for i in range( 0, 10 ):
        moose.setClock( i, 10000 )
    for i in range( 10, 18 ):
        moose.setClock( i, dt )
    moose.setClock( 18, plotdt )
    moose.reinit()
    buildPlots()
    # Run for baseline, tetanus, and post-tetanic settling time 
    print 'starting...'
    t1 = time.time()
    moose.start( baselineTime )
    caPsd = moose.vec( '/model/chem/psd/Ca_input' )
    caDend = moose.vec( '/model/chem/dend/DEND/Ca_input' )
    castim = (numpy.random.rand( len( caPsd.concInit ) ) * 0.8 + 0.2) * psdTetCa
    caPsd.concInit = castim
    caDend.concInit = numpy.random.rand( len( caDend.concInit ) ) * 1e-3
    moose.start( tetTime )
    caPsd.concInit = basalCa
    caDend.concInit = basalCa
    moose.start( interTetTime )
    caPsd.concInit = castim
    caDend.concInit = numpy.random.rand( len( caDend.concInit ) ) * 1e-3
    moose.start( tetTime )
    caPsd.concInit = basalCa
    caDend.concInit = basalCa
    moose.start( postTetTime )
    caPsd.concInit = ltdCa
    caDend.concInit = ltdCa
    moose.start( ltdTime )
    caPsd.concInit = basalCa
    caDend.concInit = basalCa
    moose.start( postLtdTime )
    print 'real time = ', time.time() - t1

    # displayPlots()

if __name__ == '__main__':
    main()
