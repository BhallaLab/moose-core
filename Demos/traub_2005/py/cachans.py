# cachans.py --- 
# 
# Filename: cachans.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Sat Apr 18 00:18:24 2009 (+0530)
# Version: 
# Last-Updated: Sat Mar 10 00:01:55 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 197
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
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 
# 

# Code:

import moose

from channel import ChannelBase
from numpy import where, linspace, exp, array
class CaChannel(ChannelBase):
    """This is just a place holder to maintain type information"""
    def __init__(self, name, parent, xpower=1.0, ypower=0.0, Ek=125e-3):
        if isinstance(parent, str):
            if moose.exists(parent + '/' + name):
                ChannelBase.__init__(self, name, parent, xpower, ypower)
                return
	ChannelBase.__init__(self, name, parent, xpower, ypower)
        self.Ek = Ek
        self.connected_to_pool = False

#import pylab
class CaL(CaChannel):
    v = array(ChannelBase.v_array)
    alpha = 1.6e3 / (1.0 + exp(-0.072 * (v * 1e3 - 5)))
    v = v + 8.9e-3
    beta = where( abs(v) * 1e3 < 1e-6,
                  1e3 * 0.1 * exp(-v / 5e-3),
                  1e3 * 0.02 * v * 1e3 / (exp(v / 5e-3) - 1))

    def __init__(self, name, parent):
        if isinstance(parent, str):
            if moose.exists(parent + '/' + name):
                CaChannel.__init__(self, name, parent, xpower=2.0, Ek=125e-3)
                return
        CaChannel.__init__(self, name, parent, xpower=2.0, Ek=125e-3)
	for i in range(config.ndivs + 1):
	    self.xGate.A[i] = CaL.alpha[i]
	    self.xGate.B[i] = CaL.beta[i]
        self.xGate.tweakAlpha()
        self.X = 0.0

class CaT(CaChannel):
    v = ChannelBase.v_array
    m_inf = 1 / (1 + exp( (- v - 56e-3) / 6.2e-3))
    tau_m = 1e-3 * (0.204 + 0.333 / ( exp(( v + 15.8e-3) / 18.2e-3 ) + 
                                      exp((- v - 131e-3) / 16.7e-3)))
    h_inf = 1 / (1 + exp(( v + 80e-3 ) / 4e-3))
    tau_h = where( v < -81e-3, 
                   1e-3 * 0.333 * exp( ( v + 466e-3 ) / 66.6e-3 ),
                   1e-3 * (9.32 + 0.333 * exp( ( -v - 21e-3 ) / 10.5e-3 )))

    def __init__(self, name, parent):
        if moose.exists(parent.path + '/' + name):
            CaChannel.__init__(self, name, parent, xpower=2.0, ypower=1.0)
            return
	CaChannel.__init__(self, name, parent, xpower=2.0, ypower=1.0)
	self.Ek = 125e-3
        self.X = 0.0
	for i in range(config.ndivs + 1):
	    self.xGate.A[i] = CaT.tau_m[i]
	    self.xGate.B[i] = CaT.m_inf[i]
	    self.yGate.A[i] = CaT.tau_h[i]
	    self.yGate.B[i] = CaT.h_inf[i]
	self.xGate.tweakTau()
	self.yGate.tweakTau()


class CaT_A(CaChannel):
    v = ChannelBase.v_array
    m_inf  = 1.0 / ( 1 + exp( ( - v * 1e3 - 52 ) / 7.4 ) )
    tau_m  = 1e-3 * (1 + .33 / ( exp( ( v * 1e3 + 27.0 ) / 10.0 ) + exp( ( - v * 1e3 - 102 ) / 15.0 )))
    
    h_inf  = 1 / ( 1 + exp( ( v * 1e3 + 80 ) / 5 ) )
    tau_h = 1e-3 * (28.30 + 0.33 / (exp(( v * 1e3 + 48.0)/ 4.0) + exp( ( -v * 1e3 - 407.0) / 50.0 ) ))

    def __init__(self, name, parent):
        if moose.exists(parent.path + '/' + name):
            CaChannel.__init__(self, name, parent, xpower=2.0, ypower=1.0, Ek=125e-3)
            return
        CaChannel.__init__(self, name, parent, xpower=2.0, ypower=1.0, Ek=125e-3)
        self.Ek = 125e-3
	for i in range(config.ndivs + 1):
	    self.xGate.A[i] = CaT_A.tau_m[i]
	    self.xGate.B[i] = CaT_A.m_inf[i]
	    self.yGate.A[i] = CaT_A.tau_h[i]
	    self.yGate.B[i] = CaT_A.h_inf[i]

        self.xGate.tweakTau()
        self.yGate.tweakTau()
	# self.xGate.A.dumpFile('cata_xa.plot')
        # self.xGate.B.dumpFile('cata_xb.plot')
	# self.yGate.A.dumpFile('cata_ya.plot')
        # self.yGate.B.dumpFile('cata_yb.plot')
        
        self.X = 0


# 
# cachans.py ends here
