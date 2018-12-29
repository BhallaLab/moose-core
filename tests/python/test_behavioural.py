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
    A behavioural system is described by two equations:
        dX/dt = AX+B
        Y     = CX+D   # usually omitted.

    Each behavioural neuron/synapse will have a behavioural system associated
    with it.
    """
    eqs = ('dv/dt=I_leak/Cm', 'I_leak=gL*(EL-v)')
    b = moose.BehavNeuron( '/n1', eqs )
    assert b, 'Not created'
    moose.reinit()
    moose.start(10)
    print( 'All done' )

def test():
    sanity_test()

def main():
    test()

if __name__ == '__main__':
    main()


