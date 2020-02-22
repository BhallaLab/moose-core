# doctest: NORMALIZE_WHITESPACE
import moose
import rdesigneur as rd

def test_21_vclamp():
    """Test vclamp.

    >>> test_21_vclamp()
    Rdesigneur: Elec model has 1 compartments and 0 spines on 0 compartments.
    [array([-0.065     , -0.06496867, -0.06503101, ..., -0.065     ,
           -0.065     , -0.065     ]), array([ 0.00000000e+00, -4.46215298e-08, -2.12672343e-08, ...,
           -2.49756618e-08, -2.49756618e-08, -2.49756618e-08])]
    """
    rdes = rd.rdesigneur(
        stimList = [['soma', '1', '.', 'vclamp', '-0.065 + (t>0.1 && t<0.2) * 0.02' ]],
        plotList = [
            ['soma', '1', '.', 'Vm', 'Soma membrane potential'],
            ['soma', '1', 'vclamp', 'current', 'Soma holding current'],
        ]
    )
    rdes.buildModel()
    moose.reinit()
    moose.start( 0.3 )
    # rdes.display(block=False)
    data = []
    for t in moose.wildcardFind('/##[TYPE=Table]'):
        data.append(t.vector)
    return data

if __name__ == '__main__':
    test_21_vclamp()
