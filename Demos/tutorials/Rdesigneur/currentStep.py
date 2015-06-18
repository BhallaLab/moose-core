########################################################################
# This program is copyright (c) Upinder S. Bhalla, NCBS, 2015.
# It is licenced under the GPL 2.1 or higher.
# There is no warranty of any kind. You are welcome to make copies under 
# the provisions of the GPL.
# This programme illustrates building a panel of multiscale models to
# test neuronal plasticity in different contexts.
########################################################################
import moogli
import numpy
import time
import pylab
import moose
from moose import neuroml
from PyQt4 import Qt, QtCore, QtGui
import matplotlib.pyplot as plt
import sys
import os
from moose.neuroml.ChannelML import ChannelML
sys.path.append('/home/bhalla/moose/trunk/Demos/util')
import rdesigneur as rd

PI = 3.14159265359
useGssa = True
combineSegments = True
#elecFileNames = ( "ca1_minimal.p", )
#elecFileNames = ( "ca1_minimal.p", "h10.CNG.swc" )
#elecFileNames = ( "h10.CNG.swc", )
#elecFileNames = ( "CA1.morph.xml", "ca1_minimal.p", "VHC-neuron.CNG.swc", "h10.CNG.swc" )
elecFileNames = ( "VHC-neuron.CNG.swc", "h10.CNG.swc" )
synSpineList = []
synDendList = []
frameRunTime = 1e-2 # 1 ms
baselineTime = 0.05
currPulseTime = 0.05
postPulseTime = 0.1
runtime = baselineTime + currPulseTime + postPulseTime
# Globals for current series
injectionCurrent = -0.5e-9
deltaCurrent = 0.2e-9
maxCurrent = 0.6e-9
somaVm = []     # Each entry is a full timeseries, one for each potl.
spineVm = []

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
    cellProto = [ [ "./cells/" + elecFileNames[0], "elec" ] ]
    chanProto = [
        ['./chans/hd.xml'], \
        ['./chans/kap.xml'], \
        ['./chans/kad.xml'], \
        ['./chans/kdr.xml'], \
        ['./chans/na3.xml'], \
        ['./chans/nax.xml'], \
        ['./chans/CaConc.xml'], \
        ['./chans/Ca.xml'], \
        ['./chans/NMDA.xml'], \
        ['./chans/Glu.xml'] \
    ]
    spineProto = [ \
        ['makeSpineProto()', 'spine' ]
    ]
    chemProto = []

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
    passiveDistrib = [ 
            [ ".", "#", "RM", "2.8", "CM", "0.01", "RA", "1.5",  \
                "Em", "-58e-3", "initVm", "-65e-3" ], \
            [ ".", "#axon#", "RA", "0.5" ] \
        ]
    chanDistrib = [ \
            ["hd", "#dend#,#apical#", "Gbar", "5e-2*(1+(p*3e4))" ], \
            ["kdr", "#", "Gbar", "p < 50e-6 ? 500 : 100" ], \
            ["na3", "#soma#,#dend#,#apical#", "Gbar", "250" ], \
            ["nax", "#soma#,#axon#", "Gbar", "1250" ], \
            ["kap", "#axon#,#soma#", "Gbar", "300" ], \
            ["kap", "#dend#,#apical#", "Gbar", \
                "300*(H(100-p*1e6)) * (1+(p*1e4))" ], \
            ["Ca_conc", "#soma#,#dend#,#apical#", "tau", "0.0133" ], \
            ["kad", "#soma#,#dend#,#apical#", "Gbar", \
                "300*H(p - 100e-6)*(1+p*1e4)" ], \
                ["Ca", "#dend#,#apical#", "Gbar", "p<160e-6? 10+ p*0.25e-6 : 50" ], \
            ["Ca", "#soma#", "Gbar", "10" ], \
        ]
    spineDistrib = [ \
            ["spine", '#apical#', "spineSpacing", "20e-6", \
                "spineSpacingDistrib", "2e-6", \
                "angle", "0", \
                "angleDistrib", str( 2*PI ), \
                "size", "1", \
                "sizeDistrib", "0.5" ] \
        ]
    chemDistrib = []

    ######################################################################
    # Here we define the mappings across scales. Format:
    # sourceObj sourceField destObj destField couplingExpr [wildcard][spatialExpn]
    # where the coupling expression is anything a muParser can evaluate,
    # using the input variable x. For example: 8e-5 + 300*x
    # For now, let's use existing adaptors which take an offset and scale.
    ######################################################################
    adaptorList = []

    ######################################################################
    # Having defined everything, now to create the rdesigneur and proceed
    # with creating the model.
    ######################################################################
    
    rd.addSpineProto() # This adds a version with an LCa channel by default.

    rdes = rd.rdesigneur(
        useGssa = useGssa, \
        combineSegments = combineSegments, \
        stealCellFromLibrary = True, \
        passiveDistrib = passiveDistrib, \
        spineDistrib = spineDistrib, \
        chanDistrib = chanDistrib, \
        chemDistrib = chemDistrib, \
        cellProto = cellProto, \
        chanProto = chanProto, \
        chemProto = chemProto, \
        adaptorList = adaptorList
    )
    #spineProto = spineProto, \

    return rdes

def buildPlots( rdes ):
    if not moose.exists( '/graphs' ):
        moose.Neutral( '/graphs' )
    numPlots = 10
    vtab = moose.Table( '/graphs/VmTab' )
    moose.connect( vtab, 'requestOut', rdes.soma, 'getVm' )
    eSpineCaTab = moose.Table( '/graphs/eSpineCaTab' )
    #moose.le( '/model/elec/head50' )
    elist = moose.wildcardFind( '/model/elec/head#' )
    numSpines = len( elist )
    assert( numSpines > 0 )
    path = elist[ numSpines / 2 ].path + "/Ca_conc"
    moose.connect( eSpineCaTab, 'requestOut', path, 'getCa' )
    eSpineVmTab = moose.Table( '/graphs/eSpineVmTab' )
    moose.connect( eSpineVmTab, 'requestOut', elist[ numSpines / 2], 'getVm' )
    eSpineGkTab = moose.Table( '/graphs/eSpineGkTab' )
    path = elist[ numSpines / 2 ].path + "/NMDA"
    moose.connect( eSpineGkTab, 'requestOut', path, 'getGk' )

def saveAndClearPlots( name ):
    print 'saveAndClearPlots( ', name, ' )'
    for i in moose.wildcardFind( "/graphs/#" ):
        #print i
        #plot stuff
        i.xplot( name + '.xplot', i.name )
    moose.delete( "/graphs" )

def displayPlots():
    pylab.subplot( 2,1,1 )
    t = numpy.arange( 0, len( somaVm[0] ), 1 ) * 50e-6
    for i in somaVm:
        pylab.plot( t, i[:len(t)] )
    pylab.legend()
    pylab.title( 'somaVm' )
    pylab.subplot( 2,1,2 )
    t = numpy.arange( 0, len( spineVm[0] ), 1 ) * 50e-6
    for i in spineVm:
        pylab.plot( t, i[:len(t)] )
    pylab.legend()
    pylab.title( 'spineVm' )
    pylab.show()

def printPsd( name ):
    # Print the vol, the path dist from soma, the electrotonic dist, and N
    psdR = moose.vec( '/model/chem/psd/tot_PSD_R' )
    neuronVoxel = moose.element( '/model/chem/spine' ).neuronVoxel
    elecComptMap = moose.element( '/model/chem/dend' ).elecComptMap
    print "len( neuronVoxel = ", len( neuronVoxel), min( neuronVoxel), max( neuronVoxel)
    print len( elecComptMap), elecComptMap[0], elecComptMap[12]
    neuron = moose.element( '/model/elec' )
    ncompts = neuron.compartments
    d = {}
    j = 0
    for i in ncompts:
        #print i
        d[i] = j
        j += 1

    f = open( name + ".txt", 'w' )
    for i in range( len( psdR ) ):
        n = psdR[i].n
        conc = psdR[i].conc
        vol = psdR[i].volume
        compt = elecComptMap[ neuronVoxel[i] ]
        #print compt
        segIndex = d[compt[0]]
        p = neuron.geometricalDistanceFromSoma[ segIndex ]
        L = neuron.electrotonicDistanceFromSoma[ segIndex ]
        s = str( i ) + "    " + str(n) + "  " + str( conc ) + "  " + str(p) + "  " + str(L) + "\n"
        f.write( s )
    f.close()

def build3dDisplay( rdes ):
    print "building 3d Display"
    app = QtGui.QApplication( sys.argv )
    compts = moose.wildcardFind( rdes.elecid.path + "/#[ISA=CompartmentBase]" )
    caElements = []
    for i in compts:
        if moose.exists( i.path + '/Ca_conc' ):
            caElements.append( moose.element( i.path + '/Ca_conc' ) )
        else:
            caElements.append( moose.element( '/library/Ca_conc' ) )
    eComptPath = map( lambda x: x.path, compts )
    morphology1 = moogli.read_morphology_from_moose( name = "", \
            path = rdes.elecid.path )
    morphology1.create_group( "group_all", eComptPath, -0.1, -0.02, \
        [0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.1] )
    viewer1 = moogli.DynamicMorphologyViewerWidget(morphology1)
    def callback1( morphology, viewer1 ):
        moose.start( frameRunTime )
        Vm = map( lambda x: moose.element( x ).Vm, compts )
        morphology.set_color( "group_all", Vm )
        currTime = moose.element( '/clock' ).currentTime
        viewer1.yaw( 0.01, 0 )
        if ( currTime < runtime ):
            deliverStim( currTime )
            return True
        displayPlots()
        return False

    viewer1.set_callback( callback1, idletime = 0 )
    viewer1.pitch( PI/2, 0 )
    viewer1.zoom( 0.4, 0 )
    viewer1.resize( 900, 900 )
    #viewer1.showMaximized()
    viewer1.show()

    app.exec_()

def deliverStim( currTime ):
    global injectionCurrent
    global somaVm
    global spineVm
    if numpy.fabs( currTime - baselineTime ) < frameRunTime/2.0 :
        # start current pulse
        elist = moose.wildcardFind( '/model/elec/#soma#' )
        assert( len( elist ) > 0 )
        elist[0].inject = injectionCurrent
        injectionCurrent += deltaCurrent
    if numpy.fabs( currTime - baselineTime - currPulseTime ) < frameRunTime/2.0 :
        # end current pulse
        elist = moose.wildcardFind( '/model/elec/#soma#' )
        assert( len( elist ) > 0 )
        elist[0].inject = 0.0
    if runtime - currTime < frameRunTime * 2.0 :
        somaVm.append( moose.element( '/graphs/VmTab' ).vector )
        spineVm.append( moose.element( '/graphs/eSpineVmTab' ).vector )
        if injectionCurrent < maxCurrent:
            print "Finished stim at: ", injectionCurrent
            moose.reinit()

def main():
    global synSpineList 
    global synDendList 
    numpy.random.seed( 1234 )
    rdes = buildRdesigneur()
    rdes.buildModel( '/model' )
    assert( moose.exists( '/model' ) )
    synSpineList = moose.wildcardFind( "/model/elec/#head#/glu,/model/elec/#head#/NMDA" )
    temp = set( moose.wildcardFind( "/model/elec/#/glu,/model/elec/#/NMDA" ) )

    synDendList = list( temp - set( synSpineList ) )
    print "num spine, dend syns = ", len( synSpineList ), len( synDendList )
    moose.reinit()
    #for i in moose.wildcardFind( '/model/elec/#apical#/#[ISA=CaConcBase]' ):
        #print i.path, i.length, i.diameter, i.parent.length, i.parent.diameter

    buildPlots( rdes )
    # Run for baseline, tetanus, and post-tetanic settling time 
    t1 = time.time()
    build3dDisplay( rdes )
    print 'real time = ', time.time() - t1

if __name__ == '__main__':
    main()
