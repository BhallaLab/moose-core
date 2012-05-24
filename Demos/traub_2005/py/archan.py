# archan.py --- 
# 
# Filename: archan.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Mon Apr 27 15:34:07 2009 (+0530)
# Version: 
# Last-Updated: Thu May 24 18:10:23 2012 (+0530)
#           By: subha
#     Update #: 19
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

import config
import moose
from numpy import exp, linspace

from channelbase import ChannelBase

class AR(ChannelBase):
    """Combined cation current."""
    v = ChannelBase.v_array
    m_inf  = 1 / ( 1 + exp( ( v * 1e3 + 75 ) / 5.5 ) )
    tau_m = 1e-3 / ( exp( -14.6 - 0.086 * v * 1e3) + exp( -1.87 + 0.07 * v * 1e3))

    def __init__(self, path, Ek=-35e-3):
	ChannelBase.__init__(self, path, 1, 0)
        self.xGate.tableA = AR.tau_m
        self.xGate.tableB = AR.m_inf
	self.xGate.tweakTau()
	self.X = 0.25
	self.Ek = Ek


def initARChannelPrototypes(libpath='/library'):
    return {'AR': AR('%s/AR' % (libpath))}

# 
# archan.py ends here
