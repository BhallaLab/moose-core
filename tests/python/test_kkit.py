# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import moose
print( '[INFO] Using moose from %s' % moose.__file__ )

scriptdir = os.path.dirname( os.path.realpath( __file__ ) )
print( 'Script dir %s' % scriptdir )


def main():
    """ This example illustrates loading, running, and saving a kinetic model
    defined in kkit format. It uses a default kkit model but you can specify
    another using the command line ``python filename runtime solver``. We use
    the gsl solver here. The model already defines a couple of plots and sets
    the runtime to 20 seconds.
    """
    solver = "gsl"  # Pick any of gsl, gssa, ee..
    mfile = os.path.join( scriptdir, 'genesis/kkit_objects_example.g' )
    runtime = 20.0
    modelId = moose.loadModel( mfile, 'model')
    moose.mooseAddChemSolver( '/model',  solver )

    # Increase volume so that the stochastic solver gssa
    # gives an interesting output
    #compt = moose.element( '/model/kinetics' )
    #compt.volume = 1e-19

    moose.reinit()
    moose.start( runtime )

    # Display all plots.
    for x in moose.wildcardFind( '/model/#graphs/conc#/#' ):
        t = np.arange( 0, x.vector.size, 1 ) * x.dt

    vals = x.vector
    stats = np.array([ vals.min(), vals.max( ), vals.mean(), vals.std( )])
    expected = np.array([0.0, 0.0004032, 0.000143, 0.0001313])
    assert np.allclose(stats, expected, atol=1e-5) , \
            'Got %s expected %s:Error %s' % (stats, expected, expected-stats )

# Run the 'main' if this script is executed standalone.
if __name__ == '__main__':
    main()
