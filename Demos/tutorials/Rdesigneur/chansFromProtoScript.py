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
sys.path.append('../../../Demos/util')
import rdesigneur as rd

PI = 3.14159265359
useGssa = True
combineSegments = True
# Pick your favourite cell here.
#elecFileName = "ca1_minimal.p"
elecFileName = "h10.CNG.swc"
#elecFileName = "CA1.morph.xml"
#elecFileName = "VHC-neuron.CNG.swc"
synSpineList = []
synDendList = []
probeInterval = 0.1
probeAmplitude = 1.0
tetanusFrequency = 100.0
tetanusAmplitude = 1000
tetanusAmplitudeForSpines = 1000
frameRunTime = 1e-3 # 1 ms
baselineTime = 0.05
tetTime = 0.1
postTetTime = 0.1
runtime = baselineTime + tetTime + postTetTime

def buildRdesigneur():
    '''
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
    '''
    cellProto = [ [ "./cells/" + elecFileName, "elec" ] ]
    chanProto = [ \
        [ 'proto20.make_K_AHP()', 'kap' ], \
        [ 'proto20.make_K_A()', 'kad' ], \
        [ 'proto20.make_K_C()', 'kca' ], \
        [ 'proto20.make_K_DR()', 'kdr' ], \
        [ 'proto20.make_Na()', 'na3' ], \
        [ 'proto20.make_Na()', 'nax' ], \
        [ 'proto20.make_Ca_conc()', 'Ca_conc' ], \
        [ 'proto20.make_Ca()', 'Ca' ], \
        [ 'proto20.make_NMDA()', 'NMDA' ], \
        [ 'proto20.make_glu()', 'glu' ], \
        [ 'proto20.make_GABA()', 'GABA' ], \
    ]
    spineProto = [ \
        ['makeSpineProto()', 'spine' ]
    ]
    chemProto = [ \
        ['./chem/' + 'psd53.g', 'ltpModel'] \
    ]

    ##################################################################
    # Here we define what goes where, and any parameters. Each distribution
    # has the format
    # protoName, path, field, expr, [field, expr]...
    # where 
    #   protoName identifies the prototype to be placed on the cell
    #   path is a MOOSE wildcard path specifying where to put things
    #   field is the field to assign.
    #   expr is a math expression to define field value. This uses the
    #     muParser. Built-in variables are:
    #       p, g, L, len, dia, maxP, maxG, maxL.
    #     where
    #       p = path distance from soma, threaded along dendrite
    #       g = geometrical distance from soma (shortest distance)
    #       L = electrotonic distance from soma: number of length constants
    #       len = length of dendritic compartment
    #       dia = diameter of dendritic compartment
    #       maxP = maximal value of 'p' for the cell
    #       maxG = maximal value of 'g' for the cell
    #       maxL = maximal value of 'L' for the cell
    #
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
            ["glu", "#dend#,#apical#", "Gbar", "400*H(p-200e-6)" ], \
            ["NMDA", "#dend#,#apical#", "Gbar", "4*H(p-200e-6)" ] \
        ]
    spineDistrib = [ \
            ["spine", '#apical#', "spineSpacing", "20e-6", \
                "spineSpacingDistrib", "2e-6", \
                "angle", "0", \
                "angleDistrib", str( 2*PI ), \
                "size", "1", \
                "sizeDistrib", "0.5" ] \
        ]
    chemDistrib = [ \
            [ "ltpModel", "#apical#", "install", "1"] 
        ]

    ######################################################################
    # Here we define the mappings across scales. Format:
    # sourceObj sourceField destObj destField offset scale
    # where the coupling expression is anything a muParser can evaluate,
    # using the input variable x. For example: 8e-5 + 300*x
    # For now, let's use existing adaptors which take an offset and scale.
    ######################################################################
    adaptorList = [
        [ 'Ca_conc', 'Ca', 'psd/Ca_input', 'concInit', 8e-5, 1 ],
        [ 'Ca_conc', 'Ca', 'dend/Ca_dend_input', 'concInit', 8e-5, 0.1 ],
        [ 'psd/tot_PSD_R', 'n', 'glu', 'Gbar', 0, 0.01 ],
    ]

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
    caPsd = moose.vec( '/model/chem/psd/Ca' )
    caHead = moose.vec( '/model/chem/spine/Ca' )
    caDend = moose.vec( '/model/chem/dend/Ca_dend_input' )
    psdR = moose.vec( '/model/chem/psd/tot_PSD_R' )
    numSpines = rdes.spineCompt.mesh.num
    assert( 2 * numSpines == len( rdes.spineComptElist ) )
    assert( len( caPsd ) == numSpines )
    assert( len( caHead ) == numSpines )
    if numSpines < numPlots:
        caPsdTab = moose.Table2( '/graphs/caPsdTab', numSpines ).vec
        caHeadTab = moose.Table2( '/graphs/caHeadTab', numSpines ).vec
        psdRtab = moose.Table2( '/graphs/psdRtab', numSpines ).vec
        for i in range( numSpines ):
            moose.connect( caPsdTab[i], 'requestOut', caPsd[i], 'getConc' )
            moose.connect( caHeadTab[i], 'requestOut', caHead[i], 'getConc')
            moose.connect( psdRtab[i], 'requestOut', psdR[i], 'getN' )
    else:
        caPsdTab = moose.Table2( '/graphs/caPsdTab', numPlots ).vec
        caHeadTab = moose.Table2( '/graphs/caHeadTab', numPlots ).vec
        psdRtab = moose.Table2( '/graphs/psdRtab', numPlots ).vec
        dx = numSpines / numPlots
        for i in range( numPlots ):
            moose.connect( caPsdTab[i], 'requestOut', caPsd[i*dx], 'getConc' )
            moose.connect( caHeadTab[i], 'requestOut', caHead[i*dx], 'getConc' )
            moose.connect( psdRtab[i], 'requestOut', psdR[i*dx], 'getN' )
    caDendTab = moose.Table2( '/graphs/caDendTab', len( caDend ) ).vec
    for i in zip( caDendTab, caDend ):
        moose.connect( i[0], 'requestOut', i[1], 'getConc' )

    vtab = moose.Table( '/graphs/VmTab' )
    moose.connect( vtab, 'requestOut', rdes.soma, 'getVm' )
    eSpineCaTab = moose.Table( '/graphs/eSpineCaTab' )
    path = rdes.spineComptElist[1].path + "/Ca_conc"
    moose.connect( eSpineCaTab, 'requestOut', path, 'getCa' )
    eSpineVmTab = moose.Table( '/graphs/eSpineVmTab' )
    moose.connect( eSpineVmTab, 'requestOut', rdes.spineComptElist[1], 'getVm' )
    eSpineGkTab = moose.Table( '/graphs/eSpineGkTab' )
    path = rdes.spineComptElist[1].path + "/NMDA"
    moose.connect( eSpineGkTab, 'requestOut', path, 'getGk' )

def saveAndClearPlots( name ):
    print 'saveAndClearPlots( ', name, ' )'
    for i in moose.wildcardFind( "/graphs/#" ):
        #print i
        #plot stuff
        i.xplot( name + '.xplot', i.name )
    moose.delete( "/graphs" )

def displayPlots():
    pylab.figure(1, figsize = (8,10 ) )
    pylab.subplot( 4,1,1)
    for i in moose.wildcardFind( "/graphs/psdRtab[]" ):
        t = numpy.arange( 0, i.vector.size, 1 ) * i.dt
        pylab.plot( t, i.vector )
    pylab.title( '# receptor in PSD' )
    pylab.subplot( 4,1,2)
    for i in moose.wildcardFind( "/graphs/caDendTab[]" ):
        t = numpy.arange( 0, i.vector.size, 1 ) * i.dt
        pylab.plot( t, i.vector )
    pylab.title( '[Ca] in dend' )
    pylab.subplot( 4,1,3)
    for i in moose.wildcardFind( "/graphs/caHeadTab[]" ):
        t = numpy.arange( 0, i.vector.size, 1 ) * i.dt
        pylab.plot( t, i.vector )
    pylab.title( '[Ca] in spine head' )
    pylab.subplot( 4,1,4)
    for i in moose.wildcardFind( "/graphs/#VmTab" ):
        t = numpy.arange( 0, i.vector.size, 1 ) * i.dt
        pylab.plot( t, i.vector, label = i.name )
    pylab.xlabel( "Time (s)" )
    pylab.legend()
    pylab.title( 'Vm' )

    pylab.figure(2, figsize= (8,10))
    ax = pylab.subplot( 1,1,1 )
    neuron = moose.element( '/model/elec' )
    comptDistance = dict( zip( neuron.compartments, neuron.pathDistanceFromSoma ) )
    for i in moose.wildcardFind( '/library/#[ISA=ChanBase]' ):
        chans = moose.wildcardFind( '/model/elec/#/' + i.name )
        print i.name, len( chans )
        p = [ 1e6*comptDistance.get( j.parent, 0) for j in chans ]
        Gbar = [ j.Gbar/(j.parent.length * j.parent.diameter * PI) for j in chans ]
        if len( p ) > 2:
            pylab.plot( p, Gbar, linestyle = 'None', marker = ".", label = i.name )
    ax.set_yscale( 'log' )
    pylab.xlabel( "Distance from soma (microns)" )
    pylab.ylabel( "Channel density (Seimens/sq mtr)" )
    pylab.legend()
    pylab.title( 'Channel distribution' )
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
    morphology1.create_group( "group_all", eComptPath, -0.08, 0.02, \
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

    morphology2 = moogli.read_morphology_from_moose( name = "", \
            path = rdes.elecid.path )
    morphology2.create_group( "group_all", eComptPath, 0.0, 0.002, \
        [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.1] )
    viewer2 = moogli.DynamicMorphologyViewerWidget(morphology2)
    def callback2( morphology, viewer2 ):
        Ca = map( lambda x: moose.element( x ).Ca, caElements )
        morphology.set_color( "group_all", Ca )
        viewer2.yaw( 0.01, 0 )
        return True

    viewer2.set_callback( callback2, idletime = 0 )
    viewer2.pitch( PI/2, 0 )
    viewer2.zoom( 0.4, 0 )
    viewer2.resize( 900, 900 )
    #viewer1.showMaximized()
    viewer2.show()
    app.exec_()

def deliverStim( currTime ):
    if currTime > baselineTime and currTime < baselineTime + tetTime:
        # deliver tet stim
        step = int ( (currTime - baselineTime) / frameRunTime )
        tetStep = int( 1.0 / (tetanusFrequency * frameRunTime ) )
        if step % tetStep == 0:
            for i in synDendList:
                i.activation( tetanusAmplitude )
            for i in synSpineList:
                i.activation( tetanusAmplitudeForSpines )
    else:
        # deliver probe stim
        step = int (currTime / frameRunTime )
        probeStep = int( probeInterval / frameRunTime )
        if step % probeStep == 0:
            print "Doing probe Stim at ", currTime
            for i in synSpineList:
                i.activation( probeAmplitude )


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
