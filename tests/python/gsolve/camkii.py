"""camkii.py: 

Here I encode the state of CaMKII.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import moose
import moose.utils as mu


class CaMKII( ):

    def __init__(self, nP = 0, nPP1 = 0):
        """A CaMKII molecule with nP subunits phosphorylated and nPP1
        molecules of PP1 attached to phosphorylated subunits. 

        nPP1 must be less than equal to nPhospho
        """
        assert nP >= nPP1, "No of phosphorylated subunits must not be less than"
        " number of PP1 molecules attached."
        self.nP = nP
        self.nPP1 = nPP1
        self.state = (self.nP, self.nPP1)
        
        # name of the CaMKII ring. x stands of no of subunits phosphorylated, y
        # stands for no of PP1 molecules attached to phoshorylated subunits.
        self.name = 'x%sy%s' % (self.nP, self.nPP1)
        self.moosePool = None

    def create_moose_pool( self, compartment = '/', buffered = False ):
        """ Create a pool in moose 
        """
        if self.moosePool:
            mu.info("Pool already exits in moose: %s" % self.moosePool.path)
        if not buffered:
            self.moosePool = moose.Pool( '%s/%s' % (compartment, self.name))
        else:
            self.moosePool = moose.BufPool( '%s/%s' % (compartment, self.name))
        return self.moosePool

    def add_pp1( self, num = 1 ):
        return (self.nP, self.nPP1 + num)

    def __repr__(self):
        return "x%s_y%s" % (self.nP, self.nPP1)
