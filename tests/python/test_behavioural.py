# -*- coding: utf-8 -*-
"""test_behavioural.py: 

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2017-, Dilawar Singh"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
import os
import numpy as np
import moose
print( "[INFO ] Using moose from %s" % moose.__file__ )

def sanity_test():

    """
    A behavioural system is described ODE system. For a neuron it is usually
        dv/dt = f(v, ...,t) etc.

    Each behavioural neuron/synapse will have a behavioural system associated
    with it.

    all units must be in SI.
    """
    eqs = ('dv/dt=I_leak/Cm+v/tau', 'I_leak=gL*(EL-v)')
    b = moose.BehavNeuron( '/n1', eqs, gL=1e-2, EL=1e-2
            , threshold=1e-3, reset=0.0, tau=10e-3
            , verbose=True
            )
    t = moose.Table( '/n1/tab' )
    moose.connect( t, 'requestOut', b, 'getVm' )
    assert b, 'Not created'
    moose.reinit()
    moose.start(1e-3)
    print( t.vector )
    print( 'All done' )

def test():
    sanity_test()

def main():
    test()

if __name__ == '__main__':
    main()


