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
display3D = False
useGssa = True
combineSegments = True
# Pick your favourite cell here.
elecFileName = "ca1_minimal.p"
#elecFileName = "h10.CNG.swc"
#elecFileName = "CA1.morph.xml"
#elecFileName = "VHC-neuron.CNG.swc"
synSpineList = []
synDendList = []
probeInterval = 0.5
probeAmplitude = 0.5
tetanusFrequency = 100
tetanusAmplitude = 0.02
tetanusAmplitudeForSpines = 1.0
frameRunTime = 2e-3 # 1 ms
baselineTime = 2e-0
tetTime = 1.0
interTetTime = 9.0
numTet = 3
postTetTime = 20
runtime = baselineTime + tetTime + interTetTime * numTet + postTetTime

def makeSpineProtoWithLCa( name ):
    rd.addSpineProto( name = name, \
        synList = ( ['glu', 0.0, 2e-3, 9e-3, 200, False],
        ['NMDA', 0.0, 20e-3, 20e-3, 20, True] ),
        chanList = (),
    )

rd.makeSpineProtoWithLCa = makeSpineProtoWithLCa

def buildRdesigneur():
    '''
    This tutorial uses Rdesigneur to make a 36-compartment neuronal model 
    with 14 spines, each having a chemical network that is switched on by
    Ca influx. The simulation delivers a 3-pulse tetanic stimulus to 
    trigger the chemical pathways.
    This simulation takes about 2 minutes to run 50 seconds of simulation time.
    Any of a variety of more detailed cell morphologies could be used for the 
    model, but they will take much longer to run.
    The flag 'display3D' decides whether the 3-D graphics of the cell will
    be shown as the simulation runs. It is likely to slow things down a lot.
    '''
    cellProto = [ [ "./cells/" + elecFileName, "elec" ] ]
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
        ['makeSpineProtoWithLCa()', 'spine' ]
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
                "30*H(p - 100e-6)*(1+p*1e4)" ], \
            ["Ca", "#dend#,#apical#", "Gbar", "p<160e-6? 10+ p*0.25e6 : 50" ], \
            ["Ca", "#soma#", "Gbar", "10" ], \
            ["glu", "#dend#,#apical#", "Gbar", "200*H(p-200e-6)" ], \
            ["NMDA", "#dend#,#apical#", "Gbar", "2*H(p-200e-6)" ] \
        ]
    spineDistrib = [ \
            ["spine", '#apical#', "spineSpacing", "100e-6", \
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
        [ 'psd/tot_PSD_R', 'n', 'glu', 'modulation', 1.0, 0.01 ],
    ]

    ######################################################################
    # Having defined everything, now to create the rdesigneur and proceed
    # with creating the model.
    ######################################################################
    
    rdes = rd.rdesigneur(
        useGssa = useGssa, \
        combineSegments = combineSegments, \
        stealCellFromLibrary = True, \
        passiveDistrib = passiveDistrib, \
        spineDistrib = spineDistrib, \
        chanDistrib = chanDistrib, \
        chemDistrib = chemDistrib, \
        spineProto = spineProto, \
        cellProto = cellProto, \
        chanProto = chanProto, \
        chemProto = chemProto, \
        adaptorList = adaptorList
    )

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
    morphology2.create_group( "group_all", eComptPath, 0.0, 0.01, \
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

def buildStimulus( rdes, synSpineList, synDendList ):
    '''
    This is slice, so we want a regular probe stimulus plus tetanic
    stimuli as per specs. We have to send in two different intensities,
    each to the spines and dendrites. These
    are modeled as coming in at two different sets of axons. The
    firing of the stimuli is controlled by a pulseGen.
    '''
    stim = moose.Neutral( '/stim' )
    probeTimer = moose.PulseGen( '/stim/probeTimer' )
    probeTimer.baseLevel = -1
    probeTimer.firstLevel = 1
    probeTimer.firstWidth = 0.001
    probeTimer.firstDelay = probeInterval
    probeTimer.trigMode = 0

    ## Just draw out the vector values, 1 per second.
    vec = [0.0]*int(baselineTime)
    vec.extend( [2.0] )
    vec.extend( [0.0] * int(interTetTime ) )
    vec.extend( [2.0] )
    vec.extend( [0.0] * int(interTetTime ) )
    vec.extend( [2.0] )
    vec.extend( [0.0] )
    tetController = moose.StimulusTable( '/stim/tetController' )
    tetController.startTime = baselineTime
    tetController.stopTime = baselineTime + len( vec )
    tetController.stepSize = 0
    tetController.doLoop = 0
    tetController.vector = vec

    tetTimer = moose.PulseGen( '/stim/tetTimer' )
    tetTimer.baseLevel = -1
    tetTimer.firstLevel = 1
    tetTimer.firstWidth = 0.001
    tetTimer.firstDelay = 1.0/tetanusFrequency
    tetTimer.trigMode = 2
    spineStim = moose.SpikeGen( '/stim/spineStim' )
    spineStim.threshold = 0.0
    spineProbeStim = moose.SpikeGen( '/stim/spineProbeStim' )
    spineProbeStim.threshold = 0.0
    dendStim = moose.SpikeGen( '/stim/dendStim' )
    dendStim.threshold = 0.0
    moose.connect( probeTimer, 'output', spineProbeStim, 'Vm' )
    moose.connect( tetController, 'output', tetTimer, 'input' )
    moose.connect( tetTimer, 'output', spineStim, 'Vm' )
    moose.connect( tetTimer, 'output', dendStim, 'Vm' )
    for i in synSpineList:
        sh = moose.element( i.path + '/sh' )
        sh.numSynapses = 2
        sh.synapse[0].weight = tetanusAmplitudeForSpines
        sh.synapse[1].weight = probeAmplitude
        moose.connect( spineStim, 'spikeOut', sh.synapse[0], 'addSpike' )
        moose.connect( spineProbeStim, 'spikeOut',sh.synapse[1], 'addSpike')
    for i in synDendList:
        sh = moose.element( i.path + '/handler' )
        sh.numSynapses = 1
        sh.synapse[0].weight = tetanusAmplitude
        moose.connect( dendStim, 'spikeOut', sh.synapse[0], 'addSpike' )


def main():
    global synSpineList 
    global synDendList 
    numpy.random.seed( 1234 )
    rdes = buildRdesigneur()
    rdes.buildModel( '/model' )
    assert( moose.exists( '/model' ) )

    moose.le( '/model/elec/head0' )
    print moose.element( '/model/elec/head0/NMDA' ).Gbar

    synSpineList = moose.wildcardFind( "/model/elec/#head#/glu,/model/elec/#head#/NMDA" )
    temp = set( moose.wildcardFind( "/model/elec/#/glu,/model/elec/#/NMDA" ) )

    synDendList = list( temp - set( synSpineList ) )
    print "num spine, dend syns = ", len( synSpineList ), len( synDendList )
    buildStimulus( rdes, synSpineList, synDendList )
    moose.reinit()
    '''
    for i in moose.wildcardFind( '/model/elec/#head#/glu' ):
        print i.path, i.Gbar, moose.element( i.path + '/sh/synapse[0]').weight
    '''
    buildPlots( rdes )
    # Run for baseline, tetanus, and post-tetanic settling time 
    t1 = time.time()
    if display3D:
        build3dDisplay( rdes )
    else:
        moose.start( runtime )
        print 'real time = ', time.time() - t1
        displayPlots()

if __name__ == '__main__':
    main()
