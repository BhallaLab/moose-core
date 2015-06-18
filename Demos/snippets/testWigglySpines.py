##################################################################
## This program is part of 'MOOSE', the
## Messaging Object Oriented Simulation Environment.
##           Copyright (C) 2015 Upinder S. Bhalla. and NCBS
## It is made available under the terms of the
## GNU Lesser General Public License version 2.1
## See the file COPYING.LIB for the full notice.
##
## testRdesigneur.py: Builds a spiny compartment and populates it with
## a molecule that diffuses.
##################################################################

import math
import pylab
import numpy
import matplotlib.pyplot as plt
import moose
import sys
sys.path.append( '../util' )
import rdesigneur as rd
from PyQt4 import QtGui
import moogli

RM = 1.0
RA = 1.0
CM = 0.01
runtime = 1.0
diffConst = 1e-12
dendLen = 10e-6
segLen = dendLen / 5
spineSpacing = 2.0e-6
spineSpacingDistrib = 0.0
spineSize = 1.0
spineSizeDistrib = 0
spineAngle= 0.0
spineAngleDistrib = 2*numpy.pi


def makeCellProto( name ):
    elec = moose.Neuron( '/library/' + name )
    ecompt = []
    ec = rd.buildCompt( elec, 'dend', dendLen, 2.0e-6, 0, RM, RA, CM )
    elec.buildSegmentTree()

def makeChemProto( name ):
    chem = moose.Neutral( '/library/' + name )
    for i in ( 'dend', 'spine', 'psd' ):
        print 'making ', i
        compt = moose.CubeMesh( chem.path + '/' + i )
        compt.volume = 1e-18
        Ca = moose.BufPool( compt.path + '/Ca' )
        CaM = moose.Pool( compt.path + '/CaM' )
        CaCaM = moose.Pool( compt.path + '/CaCaM' )
        reac = moose.Reac( compt.path + '/reac' )
        moose.connect( reac, 'sub', Ca, 'reac' )
        moose.connect( reac, 'sub', CaM, 'reac' )
        moose.connect( reac, 'prd', CaCaM, 'reac' )
        Ca.concInit = 0.08
        CaM.concInit = 1
        CaCaM.concInit = 0
        reac.Kf = 1
        reac.Kb = 1

def makeModel():
    moose.Neutral( '/library' )
    makeCellProto( 'cellProto' )
    makeChemProto( 'cProto' )
    rdes = rd.rdesigneur( useGssa = False, \
            combineSegments = False, \
            stealCellFromLibrary = True, \
            meshLambda = 1e-6, \
            cellProto = [['cellProto', 'elec' ]] ,\
            spineProto = [['makeSpineProto()', 'spine' ]] ,\
            chemProto = [['cProto', 'chem' ]] ,\
            spineDistrib = [ \
                ['spine', '#', \
                'spacing', str( spineSpacing ), \
                'spacingDistrib', str( spineSpacingDistrib ), \
                'angle', str( spineAngle ), \
                'angleDistrib', str( spineAngleDistrib ), \
                'size', str( spineSize ), \
                'sizeDistrib', str( spineSizeDistrib ) ] \
            ], \
            chemDistrib = [ \
                [ "chem", "#", "install", "1" ] \
            ],
            adaptorList = [ \
                [ 'psd/Ca', 'conc', '.', 'inject', False, 0, 2e-9 ], \
                ] \
        )
    rdes.buildModel( '/model' )

def p2(x):
    '''
    Print to 2 sig fig
    '''
    print round(x, 1-int(numpy.floor(numpy.log10(x)))),

def printChemParms( when ):
    print "CHEM PARAMS: ", when, ":"
    print "\nPSD: ",
    for i in ( 'Ca', 'CaM' ):
        vp = moose.vec( '/model/chem/psd/' + i )
        for j in range( 5 ):
            p2( vp[j].nInit )
    print "\nHEAD: ",
    for i in ( 'Ca', 'CaM' ):
        vs = moose.vec( '/model/chem/spine/' + i )
        for j in range( 5 ):
            p2( vs[j].nInit )
    vp = moose.vec( 'model/chem/psd/reac' )
    vs = moose.vec( 'model/chem/spine/reac' )
    print "\nnumKf = ", vs.numKf, "   ", vp.numKf
    print "\nnumKb = ", vs.numKb, "   ", vp.numKb

    print "ELEC PARAMS: ", when, ":"
    print "\nHead Rm = ",
    for i in range(5 ):
        p2( moose.element( '/model/elec/head' + str(i) ).Rm )
    print "\nHead Ra = ",
    for i in range(5 ):
        p2( moose.element( '/model/elec/head' + str(i) ).Ra )
    print "\nHead Cm = ",
    for i in range(5 ):
        p2( moose.element( '/model/elec/head' + str(i) ).Cm )

    print "\nShaft Rm = ",
    for i in range(5 ):
        p2( moose.element( '/model/elec/shaft' + str(i) ).Rm )
    print "\nShaft Ra = ",
    for i in range(5 ):
        p2( moose.element( '/model/elec/shaft' + str(i) ).Ra )
    print "\nShaft Cm = ",
    for i in range(5 ):
        p2( moose.element( '/model/elec/shaft' + str(i) ).Cm )
    print

def main():
    """
    This illustrates the use of rdesigneur to build a simple dendrite with
    spines, and then to resize them using spine fields. These are the 
    fields that would be changed dynamically in a simulation with reactions
    that affect spine geometry.
    """
    makeModel()
    elec = moose.element( '/model/elec' )
    elec.setSpineAndPsdMesh( moose.element('/model/chem/spine'), moose.element('/model/chem/psd') )
    print 'Num Spines = ', elec.numSpines
    head0 = moose.element( '/model/elec/head0' )
    shaft0 = moose.element( '/model/elec/shaft0' )
    printChemParms( "before" )
    elec.spine[0].headLength = 5e-6 # 10x length
    elec.spine[1].headDiameter = 5e-6 # 10 x dia
    elec.spine[2].shaftLength = 2e-6 # 2 x length
    elec.spine[3].shaftDiameter = 1e-6 # 5 x dia
    elec.spine[4].totalLength = 3e-6 # 2 x total length
    printChemParms( "after" )

    app = QtGui.QApplication(sys.argv)
    #widget = mv.MoogliViewer( '/model' )
    morphology = moogli.read_morphology_from_moose( name="", path = '/model/elec' )
    widget = moogli.MorphologyViewerWidget( morphology )
    widget.show()
    return app.exec_()
    quit()

# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
	main()
