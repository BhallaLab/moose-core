# This example demonstrates insertion of endo-compartments into the
# dendrite. Here endo_compartments are used for the endoplasmic reticulum 
# (ER) in a model of Calcium Induced Calcium Release through the 
# IP3 receptor. It generates a series of propagating waves of calcium.
# Note that units of permeability in the ConcChan are 1/(millimolar.sec)
# 
# Copyright (C) Upinder S. Bhalla NCBS 2018
# Released under the terms of the GNU Public License V3.
# Converted to a doctest by Dilawar Singh

import moose
import numpy as np
import rdesigneur as rd

np.set_printoptions(precision=3)

def test():
    """Test CICR

    >>> test() # doctest: +NORMALIZE_WHITESPACE
    Rdesigneur: Elec model has 1 compartments and 0 spines on 1 compartments.
        Chem part of model has the following compartments: 
        | In dend, 10 voxels X 15 pools
        | In dend_endo, 10 voxels X 2 pools
    array([1.569e-06, 5.800e-13, 1.991e-08, 4.094e-01, 5.288e-04, 5.188e-04,
           5.242e-04, 5.275e-04, 5.276e-04, 5.276e-04, 5.276e-04, 5.276e-04,
           5.276e-04, 5.276e-04, 5.288e-04, 5.188e-04, 5.242e-04, 5.275e-04,
           5.276e-04, 5.276e-04, 5.276e-04, 5.276e-04, 5.276e-04, 5.276e-04,
           4.032e-01, 4.045e-01, 4.038e-01, 4.038e-01, 4.038e-01, 4.038e-01,
           4.038e-01, 4.038e-01, 4.038e-01, 4.038e-01, 1.109e-06, 1.017e-06,
           1.079e-06, 1.085e-06, 1.085e-06, 1.085e-06, 1.085e-06, 1.085e-06,
           1.085e-06, 1.085e-06])
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
        chemProto = [['./chem/CICRwithConcChan.g', 'chem']],
        chemDistrib = [['chem', 'soma', 'install', '1' ]],
        plotList = [
            ['soma', '1', 'dend/CaCyt', 'conc', 'Dendritic Ca'],
            ['soma', '1', 'dend/CaCyt', 'conc', 'Dendritic Ca', 'wave'],
            ['soma', '1', 'dend_endo/CaER', 'conc', 'ER Ca'],
            ['soma', '1', 'dend/ActIP3R', 'conc', 'active IP3R'],
        ],
    )
    rdes.buildModel()
    IP3 = moose.element( '/model/chem/dend/IP3' )
    IP3.vec.concInit = 0.004
    IP3.vec[0].concInit = 0.02
    moose.reinit()
    moose.start(20)
    return np.mean([t.vector[::20] for t in
        moose.wildcardFind('/##[TYPE=Table2]')], axis=1)

if __name__ == '__main__':
    test()
