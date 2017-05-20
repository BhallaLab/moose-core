import moose
import moose.utils as mu
import numpy as np
import matplotlib.pyplot as plt

print( '[INFO] Using moose from %s' % moose.__file__ )

tables_ = { }

def create_cylinder( name ):
    moose.Neutral( '/model' )
    c = moose.CylMesh( '/model/%s' % name )
    c.x1 = 1e-3
    c.diffLength = c.x1 / 2.0
    c.r0 = c.r1 = 1e-3
    print( 'Volume of cylinder %s is %g' % (c.name,  c.volume) )
    return c

def setup_solver( c ):
    s = moose.Stoich( '/model/stoich')
    k = moose.Ksolve( '/model/ksolve' )
    d = moose.Dsolve( '/model/dsolve' )
    # s.dsolve =  d
    s.ksolve = k
    s.compartment = c
    s.path = '%s/##' % c.path 

def make_model( ):
    c = create_cylinder( 'a' )
    pools = { }
    for x in [ 'A', 'C' ]:
        p = moose.Pool( '/model/a/%s' % x )
        t = moose.Table2( '/model/a/tab%s' %  x)
        moose.connect( t, 'requestOut', p, 'getN' )
        pools[ x ] = p
        tables_[ x ] = t

    pools[ 'A' ].concInit = 1.0 
    # pools[ 'A' ].diffConst = 1.0 
    pools[ 'C' ].concInit = 0.1 

    # A + 2B <-- kf, kb --> C + D 
    r1 = moose.Reac( '/model/a/r1' )
    moose.connect( r1, 'sub', pools['A'], 'reac' )
    # moose.connect( r1, 'sub', pools['B'], 'reac' )
    # moose.connect( r1, 'sub', pools['B'], 'reac' )
    moose.connect( r1, 'prd', pools['C'], 'reac' )
    # moose.connect( r1, 'prd', pools['D'], 'reac' )
    r1.Kf = 1
    r1.Kb = 0.1

    setup_solver( c )
    moose.reinit( )   # Build the system
    moose.start( 1 ) # Run the system
    mu.plot_records( tables_, subplot = True, outfile = 'result.png' )
    k = moose.element( '/model/ksolve' )
    print( k.numPools )


def main( ):
    make_model( )

if __name__ == '__main__':
    main()
