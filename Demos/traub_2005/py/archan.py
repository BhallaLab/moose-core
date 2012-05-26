# archan.py --- 
# 
# Filename: archan.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Mon Apr 27 15:34:07 2009 (+0530)
# Version: 
# Last-Updated: Sat May 26 12:25:11 2012 (+0530)
#           By: subha
#     Update #: 33
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 
# 
# 
# 
# 

# Code:

from numpy import exp
import moose
from channelbase import ChannelBase

class AR(ChannelBase):
    """Combined cation current."""
    _prototypes = {}
    v = ChannelBase.v_array
    m_inf  = 1 / ( 1 + exp( ( v * 1e3 + 75 ) / 5.5 ) )
    tau_m = 1e-3 / ( exp( -14.6 - 0.086 * v * 1e3) + exp( -1.87 + 0.07 * v * 1e3))

    def __init__(self, path, Ek=-35e-3):
        if moose.exists(path):
            ChannelBase.__init__(self, path, xpower=1, ypower=0)
            return
        ChannelBase.__init__(self, path, xpower=1, ypower=0)
        self.xGate.tableA = AR.tau_m
        self.xGate.tableB = AR.m_inf
	self.xGate.tweakTau()
	self.X = 0.25
	self.Ek = Ek


def initARChannelPrototypes(libpath='/library'):
    if AR._prototypes:
        return AR._prototypes
    path = '%s/AR' % (libpath)
    AR._prototypes['AR'] = AR(path)
    print 'Created channel prototype:', path
    return AR._prototypes

# 
# archan.py ends here
