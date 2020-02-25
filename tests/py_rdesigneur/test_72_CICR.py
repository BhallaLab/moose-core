# This example demonstrates insertion of endo-compartments into the
# dendrite. Here endo_compartments are used for the endoplasmic reticulum
# (ER) in a model of Calcium Induced Calcium Release through the
# IP3 receptor. It generates a series of propagating waves of calcium.
# Note that units of permeability in the ConcChan are 1/(millimolar.sec)
#
# Copyright (C) Upinder S. Bhalla NCBS 2018
# Released under the terms of the GNU Public License V3.
# Converted to a test by Dilawar Singh, 2020

import matplotlib as mpl
mpl.use('Agg')

import os
import moose
print("[INFO ] Using moose from %s (%s)" % (moose.__file__, moose.version()))

import numpy as np
import rdesigneur as rd

np.set_printoptions(precision=3)

sdir_ = os.path.dirname(os.path.realpath(__file__))

E = (np.array([
    1.090e-07, 7.281e-13, 2.754e-08, 4.094e-01, 1.733e-05, 1.733e-05,
    1.733e-05, 1.733e-05, 1.733e-05, 1.733e-05, 1.733e-05, 1.733e-05,
    1.733e-05, 1.733e-05, 1.733e-05, 1.733e-05, 1.733e-05, 1.733e-05,
    1.733e-05, 1.733e-05, 1.733e-05, 1.733e-05, 1.733e-05, 1.733e-05,
    4.089e-01, 4.089e-01, 4.089e-01, 4.089e-01, 4.089e-01, 4.089e-01,
    4.089e-01, 4.089e-01, 4.089e-01, 4.089e-01, 0.000e+00, 0.000e+00,
    0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
    0.000e+00, 0.000e+00
]),
     np.array([
         2.648e-06, 3.539e-12, 1.063e-07, 2.596e-05, 2.022e-06, 2.024e-06,
         2.024e-06, 2.024e-06, 2.024e-06, 2.024e-06, 2.024e-06, 2.024e-06,
         2.024e-06, 2.024e-06, 2.022e-06, 2.024e-06, 2.024e-06, 2.024e-06,
         2.024e-06, 2.024e-06, 2.024e-06, 2.024e-06, 2.024e-06, 2.024e-06,
         2.403e-05, 2.292e-05, 2.291e-05, 2.291e-05, 2.291e-05, 2.291e-05,
         2.291e-05, 2.291e-05, 2.291e-05, 2.291e-05, 0.000e+00, 0.000e+00,
         0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
         0.000e+00, 0.000e+00
     ]))


def test():
    """Test CICR
    """
    rdes = rd.rdesigneur(
        turnOffElec=True,
        chemDt=0.005,
        chemPlotDt=0.02,
        numWaveFrames=200,
        diffusionLength=1e-6,
        useGssa=False,
        addSomaChemCompt=False,
        addEndoChemCompt=True,
        # cellProto syntax: ['somaProto', 'name', dia, length]
        cellProto=[['somaProto', 'soma', 2e-6, 10e-6]],
        chemProto=[[os.path.join(sdir_, 'chem', 'CICRwithConcChan.g'),
                    'chem']],
        chemDistrib=[['chem', 'soma', 'install', '1']],
        plotList=[
            ['soma', '1', 'dend/CaCyt', 'conc', 'Dendritic Ca'],
            ['soma', '1', 'dend/CaCyt', 'conc', 'Dendritic Ca', 'wave'],
            ['soma', '1', 'dend_endo/CaER', 'conc', 'ER Ca'],
            ['soma', '1', 'dend/ActIP3R', 'conc', 'active IP3R'],
        ],
    )
    rdes.buildModel()
    IP3 = moose.element('/model/chem/dend/IP3')
    IP3.vec.concInit = 0.004
    IP3.vec[0].concInit = 0.02
    moose.reinit()
    k = moose.wildcardFind('/##[TYPE=Ksolve]')[0]

    moose.start(20)
    data = [t.vector for t in moose.wildcardFind('/##[TYPE=Table2]')]
    m, s = np.mean(data, axis=1), np.std(data, axis=1)
    # print(np.array_repr(m))
    # print(np.array_repr(s))
    # In multithreaded mode, the numers are not exactly the same as in
    # expected.
    assert np.allclose(m, E[0], rtol=1e-3), (m, E[0])
    assert np.allclose(s, E[1], rtol=1e-3), (s, E[1])
    print('done')


if __name__ == '__main__':
    test()
