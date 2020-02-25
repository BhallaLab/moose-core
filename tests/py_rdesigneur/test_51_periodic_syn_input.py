# This example demonstrates periodic synaptic input to a cell.
# Copyright (C) Upinder S. Bhalla NCBS 2018
# Released under the terms of the GNU Public License V3. No warranty.
# Doctest by Dilawar Singh

import moose
import rdesigneur as rd
import numpy as np

np.set_printoptions(precision=3)

def test():
    """Test periodic input.
    >>> test()
    Rdesigneur: Elec model has 1 compartments and 0 spines on 0 compartments.
    array([-0.065, -0.065, -0.065, -0.065, -0.065, -0.065, -0.065, -0.065,
           -0.065, -0.065, -0.065, -0.062, -0.057, -0.054, -0.052, -0.051,
           -0.051, -0.051, -0.052, -0.053, -0.054, -0.053, -0.05 , -0.048,
           -0.047, -0.047, -0.047, -0.048, -0.05 , -0.051, -0.052, -0.051,
           -0.049, -0.047, -0.046, -0.046, -0.047, -0.048, -0.049, -0.051,
           -0.052, -0.051, -0.048, -0.047, -0.046, -0.046, -0.047, -0.048,
           -0.049, -0.05 , -0.052, -0.051, -0.048, -0.047, -0.046, -0.046,
           -0.047, -0.048, -0.049, -0.05 , -0.052, -0.051, -0.048, -0.047,
           -0.046, -0.046, -0.047, -0.048, -0.049, -0.05 , -0.052, -0.051,
           -0.048, -0.047, -0.046, -0.046, -0.047, -0.048, -0.049, -0.05 ,
           -0.052, -0.051, -0.049, -0.047, -0.046, -0.046, -0.047, -0.048,
           -0.049, -0.05 , -0.052, -0.051, -0.049, -0.047, -0.046, -0.046,
           -0.047, -0.048, -0.049, -0.05 , -0.052, -0.051, -0.049, -0.047,
           -0.046, -0.046, -0.047, -0.048, -0.049, -0.05 , -0.052, -0.051,
           -0.049, -0.047, -0.046, -0.046, -0.047, -0.048, -0.049, -0.05 ,
           -0.052, -0.051, -0.049, -0.047, -0.046, -0.046, -0.047, -0.048,
           -0.049, -0.05 , -0.052, -0.051, -0.049, -0.047, -0.046, -0.046,
           -0.047, -0.048, -0.049, -0.05 , -0.052, -0.051, -0.049, -0.047,
           -0.046, -0.046, -0.047, -0.048, -0.049, -0.05 , -0.052])
    """
    rdes = rd.rdesigneur(
        cellProto = [['somaProto', 'soma', 20e-6, 200e-6]],
        chanProto = [['make_glu()', 'glu']],
        chanDistrib = [['glu', 'soma', 'Gbar', '1' ]],

        # Deliver stimulus to glu synapse on soma, periodically at 50 Hz.
        stimList = [['soma', '0.5', 'glu', 'periodicsyn', '50' ]],
        plotList = [['soma', '1', '.', 'Vm', 'Soma membrane potential']]
    )
    rdes.buildModel()
    moose.reinit()
    moose.start( 0.3 )
    t = moose.wildcardFind('/##[TYPE=Table]')[0].vector
    expected = [-0.04995514162861773, 0.004795008283676097]
    assert np.allclose(expected, [t.mean(), t.std()]), \
            (t.mean(), t.std())

    return t

if __name__ == '__main__':
    test()
