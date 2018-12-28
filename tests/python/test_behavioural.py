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
    b = moose.BehaviouralNeuron( '/n1' )
    assert b, 'Not created'
    vs = ('a', 'tau')
    eqs = ('dv/dt=I_leak/Cm', 'I_leak=gL*(EL-v)')
    b.variables = vs
    b.equations = eqs
    assert b.variables  == vs, 'Expected %s got %s' % (vs, b.variables)
    assert b.equations  == eqs, 'Expected %s got %s' % (eqs, b.equations)
    moose.reinit()
    moose.start(10)
    print( 'All done' )

def test():
    sanity_test()

def main():
    test()

if __name__ == '__main__':
    main()


