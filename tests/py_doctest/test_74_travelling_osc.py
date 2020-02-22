# This example illustrates molecular transport of an oscillatory reaction
# system, along a closed-end cylinder. The concentrations along the entire 
# length of the cylinder start out uniform. It shows an interesting 
# combination of propagating oscillations and buildup of reagents.
# Author: Upinder Bhalla 
# Converted to pydoc test by Dilawar Singh

import moose
import numpy as np
import pylab
import rdesigneur as rd

np.set_printoptions(precision=3)

def test():
    """Test travelling wave.

    >>> test() # doctest: +NORMALIZE_WHITESPACE
    Rdesigneur: Elec model has 1 compartments and 0 spines on 1 compartments.
        Chem part of model has the following compartments: 
        | In dend, 250 voxels X 3 pools
    array([0.1  , 0.155, 0.14 , 0.128, 0.148, 0.133, 0.138, 0.141, 0.135,
           0.143, 0.137, 0.143, 0.139, 0.144, 0.14 , 0.144, 0.142, 0.143,
           0.143, 0.143, 0.146, 0.143, 0.156, 0.144, 0.156, 0.145, 0.152,
           0.147, 0.148, 0.15 , 0.148, 0.162, 0.149, 0.161, 0.15 , 0.155,
           0.152, 0.152, 0.191, 0.154, 0.155])
    """
    rdes = rd.rdesigneur(
        turnOffElec = True,
        diffusionLength = 2e-6, 
        chemProto = [['makeChemOscillator()', 'osc']],
        chemDistrib = [['osc', 'soma', 'install', '1' ]],
        plotList = [
            ['soma', '1', 'dend/a', 'conc', 'Concentration of a'],
            ['soma', '1', 'dend/b', 'conc', 'Concentration of b'],
            ['soma', '1', 'dend/a', 'conc', 'Concentration of a', 'wave'],
        ],
        moogList = [['soma', '1', 'dend/a', 'conc', 'a Conc', 0, 360 ]]
    )
    a = moose.element( '/library/osc/kinetics/a' )
    b = moose.element( '/library/osc/kinetics/b' )
    s = moose.element( '/library/osc/kinetics/s' )
    a.diffConst = 0
    b.diffConst = 0
    a.motorConst = 1e-6
    rdes.buildModel()
    moose.reinit()
    #  rdes.displayMoogli( 1, 400, rotation = 0, azim = -np.pi/2, elev = 0.0 )
    moose.start(400)
    data = []
    for t in moose.wildcardFind('/##[TYPE=Table2]'):
        data.append(t.vector[::10])
    return np.mean(data, axis=0)

if __name__ == '__main__':
    test()
