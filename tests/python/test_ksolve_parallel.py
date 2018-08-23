import numpy as np
import moose
print( 'Using moose from %s' % moose.__file__ )
import time
def main():
    """
    This example implements a reaction-diffusion like system which is
    bistable and propagates losslessly. It is based on the NEURON example 
    rxdrun.py, but incorporates more compartments and runs for a longer time.
    The system is implemented as a hybrid of a reaction and a function which
    sets its rates. Please see rxdFuncDiffusion.py for a variant that uses
    just a function object to set up the system.
    """

    dt = 0.1

    # define the geometry
    compt = moose.CylMesh( '/cylinder' )
    compt.r0 = compt.r1 = 1
    compt.x1 = 100
    compt.diffLength = 0.2
    assert( compt.numDiffCompts == compt.x1/compt.diffLength )

    #define the molecule. Its geometry is defined by its parent volume, cylinder
    c = moose.Pool( '/cylinder/pool' )
    c.diffConst = 1 # define diffusion constant
    # There is an implicit reaction substrate/product. MOOSE makes it explicit.
    buf = moose.BufPool( '/cylinder/buf' )
    buf.nInit = 1

    # The reaction is something entirely peculiar, not a chemical thing.
    reaction = moose.Reac( '/cylinder/reac' )
    reaction.Kb = 0

    # so here we set up a function calculation to do the same thing.
    func = moose.Function( '/cylinder/reac/func' )
    func.expr = "(1 - x0) * (0.3 - x0)"
    func.x.num = 1 #specify number of input variables.

    #Connect the reaction to the pools
    moose.connect( reaction, 'sub', c, 'reac' )
    moose.connect( reaction, 'prd', buf, 'reac' )

    #Connect the function to the reaction
    moose.connect( func, 'valueOut', reaction, 'setNumKf' )

    #Connect the molecules to the func
    moose.connect( c, 'nOut', func.x[0], 'input' )

    #Set up solvers
    ksolve = moose.Ksolve( '/cylinder/ksolve' )
    dsolve = moose.Dsolve( '/cylinder/dsolve' )
    stoich = moose.Stoich( '/cylinder/stoich' )
    stoich.compartment = compt
    stoich.ksolve = ksolve
    stoich.dsolve = dsolve
    stoich.path = '/cylinder/##'
    for i in range( 10, 18 ):
        moose.setClock( i, dt )

    #initialize
    x = np.arange( 0, compt.x1, compt.diffLength )
    c.vec.nInit = [ (q < 0.2 * compt.x1) for q in x ]

    expected = [ (0.2, 0.4000000000000001)
            , (0.3355471454768751, 0.45713472793392007)
            , (0.4759264630589813, 0.4852188847862358)
            , (0.6163057931092064, 0.47168830962559016)
            , (0.7566851225599798, 0.4124677536200488) 
            ]

    # Run and plot it.
    moose.reinit()
    updateDt = 50
    runtime = updateDt * 4
    #  plt = pylab.plot( x, c.vec.n, label='t = 0 ')
    yvec = c.vec.n
    u1, m1 = np.mean( yvec ), np.std( yvec )
    print( u1, m1 )
    assert np.isclose( (u1, m1), expected[0]).all()
    t1 = time.time()
    for i, t in enumerate(range( 0, runtime-1, updateDt)):
        moose.start( updateDt )
        #  plt = pylab.plot( x, c.vec.n, label='t = '+str(t + updateDt) )
        yvec = c.vec.n
        u1, m1 = np.mean( yvec ), np.std( yvec )
        print( u1, m1 )
        np.isclose( (u1,m1), expected[i+1] ).all()
    print( "Time = ", time.time() - t1 )


if __name__ == '__main__':
    main()
