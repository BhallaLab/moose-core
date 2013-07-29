# Author: Subhasis Ray
# Created: Mon Jul 29 10:47:48 IST 2013
"""This script is for tracking a segmentation fault at exit in case of
prototype channels are copied."""

import moose

def test_segfault_with_copied_element():
    lib = moose.Neutral('/library')
    proto_chan = moose.HHChannel('/library/NaChan')
    proto_chan.Xpower = 3
    proto_chan.Ypower = 1
    mgate = moose.element(proto_chan.gateX)    
    # These are dummy values
    mgate.tableA = [1, 2, 3]
    mgate.tableB = [4, 5, 6]
    hgate = moose.element(proto_chan.gateY)
    hgate.tableA = [1, 2, 3]
    hgate.tableB = [4, 5, 6]
    comp = moose.Compartment('/library/soma')
    moose.copy(proto_chan, comp, 'na')
    
if __name__ == '__main__':
    test_segfault_with_copied_element()
