import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import moose
print("[INFO ] MOOSE version=%s, loc=%s" % (moose.version(), moose.__file__))
import rdesigneur as rd

def test():
    """Test current pulse.
    >>> test() # doctest: +NORMALIZE_WHITESPACE
    Rdesigneur: Elec model has 1 compartments and 0 spines on 0 compartments.
    (array([-0.065     , -0.06468672, -0.06438269, ..., -0.0544    ,
           -0.0544    , -0.0544    ]), (array([   7,    8,   11,   17,   34, 1852,   49,   31,   34,  958]), array([-0.065     , -0.06309117, -0.06118235, -0.05927352, -0.05736469,
           -0.05545587, -0.05354704, -0.05163822, -0.04972939, -0.04782056,
           -0.04591174])))
    """
    rdes = rd.rdesigneur(
        stimList = [['soma', '1', '.', 'inject', '(t>0.1 && t<0.2) * 2e-8' ]],
        plotList = [['soma', '1', '.', 'Vm', 'Soma membrane potential']]
    )
    rdes.buildModel()
    moose.reinit()
    moose.start(0.3)
    rdes.display(block=False)

    y = moose.wildcardFind('/##[TYPE=Table]')[0].vector
    X = np.histogram(y)
    return y, X

if __name__ == '__main__':
    test()
