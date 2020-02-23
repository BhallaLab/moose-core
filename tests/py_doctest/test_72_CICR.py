# This example demonstrates insertion of endo-compartments into the
# dendrite. Here endo_compartments are used for the endoplasmic reticulum 
# (ER) in a model of Calcium Induced Calcium Release through the 
# IP3 receptor. It generates a series of propagating waves of calcium.
# Note that units of permeability in the ConcChan are 1/(millimolar.sec)
# 
# Copyright (C) Upinder S. Bhalla NCBS 2018
# Released under the terms of the GNU Public License V3.
# Converted to a doctest by Dilawar Singh, 2020 

import os
import moose
print("[INFO ] Using moose from %s (%s)" % (moose.__file__, moose.version()))

import numpy as np
import rdesigneur as rd

np.set_printoptions(precision=3)

sdir_ = os.path.dirname(os.path.realpath(__file__))

expected = (np.array([    0,     0,     0,     0,   972,   954,   981,   984,   984,
             984,   984,   984,   984,   984,   972,   954,   981,   984,
             984,   984,   984,   984,   984,   984, 95384, 95687, 95503,
           95511, 95512, 95512, 95512, 95512, 95512, 95512,     2,     1,
               1,     1,     1,     1,     1,     1,     1,     1]), 
    np.array([   0,    0,    0,    0, 2852, 2731, 2770, 2770, 2770, 2770, 2770,
           2770, 2770, 2770, 2852, 2731, 2770, 2770, 2770, 2770, 2770, 2770,
           2770, 2770, 2998, 2742, 2820, 2839, 2840, 2840, 2840, 2840, 2840,
           2840,    4,    3,    3,    3,    3,    3,    3,    3,    3,    3]))

def test():
    """Test CICR

    >>> test() # doctest: +NORMALIZE_WHITESPACE
    Rdesigneur: Elec model has 1 compartments and 0 spines on 1 compartments.
        Chem part of model has the following compartments: 
        | In dend, 10 voxels X 15 pools
        | In dend_endo, 10 voxels X 2 pools
    True
    """
    rdes = rd.rdesigneur(
        turnOffElec = True,
        chemDt = 0.005,
        chemPlotDt = 0.02,
        numWaveFrames = 200,
        diffusionLength = 1e-6,
        useGssa = False,
        addSomaChemCompt = False,
        addEndoChemCompt = True,
        # cellProto syntax: ['somaProto', 'name', dia, length]
        cellProto = [['somaProto', 'soma', 2e-6, 10e-6]],
        chemProto = [[os.path.join(sdir_, 'chem', 'CICRwithConcChan.g'), 'chem']],
        chemDistrib = [['chem', 'soma', 'install', '1' ]],
        plotList = [
            ['soma', '1', 'dend/CaCyt', 'n', 'Dendritic Ca'],
            ['soma', '1', 'dend/CaCyt', 'n', 'Dendritic Ca', 'wave'],
            ['soma', '1', 'dend_endo/CaER', 'n', 'ER Ca'],
            ['soma', '1', 'dend/ActIP3R', 'n', 'active IP3R'],
        ],
    )
    rdes.buildModel()
    IP3 = moose.element( '/model/chem/dend/IP3' )
    IP3.vec.concInit = 0.004
    IP3.vec[0].concInit = 0.02
    moose.reinit()
    k = moose.wildcardFind('/##[TYPE=Ksolve]')[0]

    moose.start(20)
    data = [t.vector for t in moose.wildcardFind('/##[TYPE=Table2]')]
    m, s = np.mean(data, axis=1).astype(int), np.std(data, axis=1).astype(int)
    # In multithreaded mode, the numers are not exactly the same as in
    # expected.
    assert abs((expected[0] - m).mean()) < 0.5, (m - expected[0])
    assert abs((expected[1] - s).mean()) < 1, (s - expected[1])
    return True

if __name__ == '__main__':
    test()
