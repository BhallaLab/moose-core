# -*- coding: utf-8 -*-

# This test shows how to properly use metaclass to avoid 
# > TypeError: metaclass conflict: the metaclass of a derived class must be 
# > a (non-strict) subclass of the metaclasses of all its bases
# error.

import moose
try:
    import six
except ImportError:
    print("[INFO ] Requires `six` to run this test.")
    quit(0)

# Can't use type here. Use metaclass of the base class (i.e.
# moose.melement).
class ChannelMeta(moose.melement.__class__):
    def __new__(cls, name, bases, cdict):     
        if  'abstract' in cdict and cdict['abstract'] == True:
            return type.__new__(cls, name, bases, cdict)


@six.add_metaclass(ChannelMeta)
class ChannelBase(moose.HHChannel):
    annotation = {'cno': 'cno_0000047'}
    abstract = True
    def __init__(self, path, xpower=1, ypower=0, Ek=0.0):
        moose.HHChannel.__init__(self, path)

def test_metaclass():
    a = ChannelBase('a')
    print(a)

def main():
    test_metaclass()

if __name__ == '__main__':
    main()
