# -*- coding: utf-8 -*-
#########################################################################
## This program is part of 'MOOSE', the
## Messaging Object Oriented Simulation Environment.
##           Copyright (C) 2013 Upinder S. Bhalla. and NCBS
## It is made available under the terms of the
## GNU Lesser General Public License version 2.1
## See the file COPYING.LIB for the full notice.
#########################################################################

import numpy
import moose

useY = False

def makeModel():
    # create container for model
    moose.Neutral( 'model' )
    compartment = moose.CubeMesh( '/model/compartment' )
    compartment.volume = 1e-15
    # the mesh is created automatically by the compartment
    moose.element( '/model/compartment/mesh' ) 

    # create molecules and reactions
    # a <----> b
    # b + 10c ---func---> d
    a = moose.Pool( '/model/compartment/a' )
    b = moose.Pool( '/model/compartment/b' )
    c = moose.Pool( '/model/compartment/c' )
    d = moose.BufPool( '/model/compartment/d' )
    reac = moose.Reac( '/model/compartment/reac' )
    func = moose.Function( '/model/compartment/d/func' )
    func.numVars = 2
    #func.x.num = 2

    # connect them up for reactions

    moose.connect( reac, 'sub', a, 'reac' )
    moose.connect( reac, 'prd', b, 'reac' )
    if useY:
        moose.connect( func, 'requestOut', b, 'getN' )
        moose.connect( func, 'requestOut', c, 'getN' )
    else:
        moose.connect( b, 'nOut', func.x[0], 'input' )
        moose.connect( c, 'nOut', func.x[1], 'input' )

    moose.connect( func, 'valueOut', d, 'setN' )
    if useY:
        func.expr = "y0 + 10*y1"
    else:
        func.expr = "x0 + 10*x1"

    # connect them up to the compartment for volumes
    #for x in ( a, b, c, cplx1, cplx2 ):
    #                        moose.connect( x, 'mesh', mesh, 'mesh' )

    # Assign parameters
    a.concInit = 1
    b.concInit = 0.5
    c.concInit = 0.1
    reac.Kf = 0.001
    reac.Kb = 0.01

    # Create the output tables
    moose.Neutral( '/model/graphs' )
    outputA = moose.Table2 ( '/model/graphs/concA' )
    outputB = moose.Table2 ( '/model/graphs/concB' )
    outputC = moose.Table2 ( '/model/graphs/concC' )
    outputD = moose.Table2 ( '/model/graphs/concD' )

    # connect up the tables
    moose.connect( outputA, 'requestOut', a, 'getConc' );
    moose.connect( outputB, 'requestOut', b, 'getConc' );
    moose.connect( outputC, 'requestOut', c, 'getConc' );
    moose.connect( outputD, 'requestOut', d, 'getConc' );

def test_func_change_expr():
    makeModel()
    ksolve = moose.Ksolve( '/model/compartment/ksolve' )
    stoich = moose.Stoich( '/model/compartment/stoich' )
    stoich.compartment = moose.element( '/model/compartment' )
    stoich.ksolve = ksolve
    stoich.path = "/model/compartment/##"
    moose.reinit()
    moose.start( 100.0 )
    func = moose.element( '/model/compartment/d/func' )
    if useY:
        func.expr = "-y0 + 10*y1"
    else:
        func.expr = "-x0 + 10*x1"
    moose.start( 100.0 ) 
    b = moose.element('/model/compartment/b')
    assert int(b.n) == int(106384558.57472235), b.n
    xs = func.x
    assert len(xs.value) == 2, (len(xs.value), xs.value)
    assert (xs.value == [0, 0]).all(), xs.value

if __name__ == '__main__':
    test_func_change_expr()
