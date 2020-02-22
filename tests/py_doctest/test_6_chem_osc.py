import moose
import numpy as np
import rdesigneur as rd

np.set_printoptions(precision=3)

def test():
    """Test chemical oscillation.

    >>> test() # doctest: +NORMALIZE_WHITESPACE
    Rdesigneur: Elec model has 1 compartments and 0 spines on 1 compartments.
        Chem part of model has the following compartments: 
        | In dend, 1 voxels X 3 pools
    [array([1.000e-01, 1.140e-05, 2.334e-03, 2.456e-01, 1.297e-01, 7.005e-02,
           2.858e-01, 5.219e-02, 2.284e-01, 1.137e-01, 1.244e-01, 2.134e-01,
           7.852e-02, 2.346e-01, 9.230e-02, 1.731e-01, 1.564e-01, 1.108e-01,
           2.136e-01, 9.126e-02, 2.043e-01]),
       array([0.5  , 0.025, 0.002, 0.117, 0.208, 0.056, 0.223, 0.093, 0.146,
           0.165, 0.092, 0.201, 0.092, 0.174, 0.131, 0.124, 0.176, 0.099,
           0.184, 0.111, 0.154])]

    """
    rdes = rd.rdesigneur(
        turnOffElec = True,
        diffusionLength = 1e-3, # Default diffusion length is 2 microns
        chemProto = [['makeChemOscillator()', 'osc']],
        chemDistrib = [['osc', 'soma', 'install', '1' ]],
        plotList = [['soma', '1', 'dend/a', 'conc', 'a Conc'],
            ['soma', '1', 'dend/b', 'conc', 'b Conc']]
    )
    rdes.buildModel()
    b = moose.element( '/model/chem/dend/b' )
    b.concInit *= 5
    moose.reinit()
    moose.start(200)
    return [t.vector[::10] for t in moose.wildcardFind('/##[TYPE=Table2]')]

if __name__ == '__main__':
    test()
