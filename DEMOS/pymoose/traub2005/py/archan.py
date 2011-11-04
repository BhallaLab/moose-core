# archan.py --- 
# 
# Filename: archan.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Mon Apr 27 15:34:07 2009 (+0530)
# Version: 
# Last-Updated: Fri Oct 21 16:04:17 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 10
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

from channel import ChannelBase

class AR(ChannelBase):
    """Combined cation current."""
    v = ChannelBase.v_array
    m_inf  = 1 / ( 1 + exp( ( v * 1e3 + 75 ) / 5.5 ) )
    tau_m = 1e-3 / ( exp( -14.6 - 0.086 * v * 1e3) + exp( -1.87 + 0.07 * v * 1e3))

    def __init__(self, name, parent, Ek=-35e-3):
	ChannelBase.__init__(self, name, parent, 1, 0)
	for i in range(len(self.xGate.A)):
	    self.xGate.A[i] = AR.tau_m[i]
	    self.xGate.B[i] = AR.m_inf[i]
	self.xGate.tweakTau()
	self.X = 0.25
	self.Ek = Ek



# 
# archan.py ends here
