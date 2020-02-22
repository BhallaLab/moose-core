# doctest: NORMALIZE_WHITESPACE
import moose
import rdesigneur as rd

def test():
    """Test current pulse.
    >>> test()
    Rdesigneur: Elec model has 1 compartments and 0 spines on 0 compartments.
    [array([-0.065     , -0.06525877, -0.06549723, ..., -0.06676325,
           -0.06676329, -0.06676332])]
    """
    rdes = rd.rdesigneur(
        chanProto = [['make_HH_Na()', 'Na'], ['make_HH_K()', 'K']],
        chanDistrib = [
            ['Na', 'soma', 'Gbar', '1200' ],
            ['K', 'soma', 'Gbar', '360' ]],
        stimList = [['soma', '1', '.', 'inject', '(t>0.1 && t<0.2) * 1e-8' ]],
        plotList = [['soma', '1', '.', 'Vm', 'Membrane potential']])
    rdes.buildModel()
    moose.reinit()
    moose.start(0.3)
    rdes.display(block=False)
    return [x.vector for x in moose.wildcardFind('/##[TYPE=Table]')]

if __name__ == '__main__':
    test()
