# kchans.py --- 
# 
# Filename: kchans.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Apr 17 23:58:49 2009 (+0530)
# Version: 
# Last-Updated: Fri Oct 21 16:10:17 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 652
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
from numpy import where, linspace, exp, arange, ones, zeros, savetxt
import config


class KChannel(ChannelBase):
    """This is a dummy base class to keep type information."""
    def __init__(self, name, parent, xpower=1, ypower=0, Ek=-95e-3):
        if config.context.exists(parent.path + '/' + name):
            ChannelBase.__init__(self, name, parent, xpower, ypower)
            return
        ChannelBase.__init__(self, name, parent, xpower, ypower)
        self.Ek = Ek


class KDR(KChannel):
    """Delayed rectifier current

    "In hippocampal pyramidal neurons, however, it has been reported have relatively slow activation, with a time to peak of some 50-100 msec and even slower inactivation. Such a slow activation would make it ill suited to participate in the repolarization of the AP.... An equation that can describe IK(DR) in cortical neurons is
    
    IK(DR) = m^3 * h * gbar_K(DR) * (Vm - EK)
    
    where m and h depend on voltage and time."
        - Johnston & Wu, Foundations of Cellular Neurophysiology (1995).

    But in Traub 2005, the equation used is:
    
    IK(DR) = m^4 * gbar_K(DR) * (Vm - EK)
    """
    v = ChannelBase.v_array
    tau_m = where(v < -10e-3, \
                      1e-3 * (0.25 + 4.35 * exp((v + 10.0e-3) / 10.0e-3)), \
                      1e-3 * (0.25 + 4.35 * exp((- v - 10.0e-3) / 10.0e-3)))
    m_inf = 1.0 / (1.0 + exp((- v - 29.5e-3) / 10e-3))
    
    def __init__(self, name, parent, Ek=-95e-3):
        if config.context.exists(parent.path + '/' + name):
            KChannel.__init__(self, name, parent, xpower, ypower)
            return
	KChannel.__init__(self, name, parent, xpower=4.0, Ek=Ek)
	for i in range(config.ndivs + 1):
            self.xGate.A[i] = KDR.tau_m[i]
            self.xGate.B[i] = KDR.m_inf[i]
	self.xGate.tweakTau()
        self.X = 0.0
	# self.xGate.A.dumpFile('kdr_xa.plot')
        # self.xGate.B.dumpFile('kdr_xb.plot')


class KDR_FS(KChannel):
    """KDR for fast spiking neurons"""
    v = ChannelBase.v_array
    m_inf = 1.0 / (1.0 + exp((- v - 27e-3) / 11.5e-3))
    tau_m =  where(v < -10e-3, \
                       1e-3 * (0.25 + 4.35 * exp((v + 10.0e-3) / 10.0e-3)), \
                       1e-3 * (0.25 + 4.35 * exp((- v - 10.0e-3) / 10.0e-3)))

    def __init__(self, name, parent, Ek=-95e-3):
        if config.context.exists(parent.path + '/' + name):
            KChannel.__init__(self, name, parent, 4, Ek=Ek)
            return
	KChannel.__init__(self, name, parent, 4, Ek=Ek)
	for i in range(config.ndivs + 1):
            self.xGate.A[i] = KDR_FS.tau_m[i]
            self.xGate.B[i] = KDR_FS.m_inf[i]
        self.xGate.tweakTau()
        self.X = 0.0
	# self.xGate.A.dumpFile('kdrfs_xa.plot')
        # self.xGate.B.dumpFile('kdrfs_xb.plot')

class KA(KChannel):
    """A type K+ channel"""
    v = ChannelBase.v_array
    m_inf = 1 / ( 1 + exp( ( - v - 60e-3 ) / 8.5e-3 ) )
    tau_m =  1e-3 * (0.185 + 0.5 / ( exp( ( v + 35.8e-3 ) / 19.7e-3 ) + exp( ( - v - 79.7e-3 ) / 12.7e-3 ) ))
    h_inf =   1 / ( 1 + exp( ( v + 78e-3 ) / 6e-3 ) )
    tau_h = where( v <= -63e-3,\
                       1e-3 * 0.5 / ( exp( ( v + 46e-3 ) / 5e-3 ) + exp( ( - v - 238e-3 ) / 37.5e-3 ) ), \
                       9.5e-3)

    def __init__(self, name, parent, Ek=-95e-3):
        if config.context.exists(parent.path + '/' + name):
            KChannel.__init__(self, name, parent, 4, 1)
            return
	KChannel.__init__(self, name, parent, 4, 1)
	for i in range(config.ndivs + 1):
            self.xGate.A[i] = KA.tau_m[i]
            self.xGate.B[i] = KA.m_inf[i]
            self.yGate.A[i] = KA.tau_h[i]
            self.yGate.B[i] = KA.h_inf[i]
        self.xGate.tweakTau()
	self.yGate.tweakTau()
        self.X = 0.0

class KA_IB(KChannel):
    """A type K+ channel for tufted intrinsically bursting cells -
    multiplies tau_h of KA by 2.6"""
    v = ChannelBase.v_array
    m_inf = 1 / ( 1 + exp( ( - v - 60e-3 ) / 8.5e-3 ) )
    tau_m =  1e-3 * (0.185 + 0.5 / ( exp( ( v + 35.8e-3 ) / 19.7e-3 ) + exp( ( - v - 79.7e-3 ) / 12.7e-3 ) ))
    h_inf =   1 / ( 1 + exp( ( v + 78e-3 ) / 6e-3 ) )
    tau_h = 2.6 * where( v <= -63e-3,\
                             1e-3 * 0.5 / ( exp( ( v + 46e-3 ) / 5e-3 ) + exp( ( - v - 238e-3 ) / 37.5e-3 ) ), \
                             9.5e-3)

    def __init__(self, name, parent, Ek=-95e-3):
        if config.context.exists(parent.path + '/' + name):
            KChannel.__init__(self, name, parent, 4, 1)
            return
	KChannel.__init__(self, name, parent, 4, 1, Ek=Ek)
        self.X = 0.0

	for i in range(config.ndivs + 1):
            self.xGate.A[i] = KA_IB.tau_m[i]
            self.xGate.B[i] = KA_IB.m_inf[i]
            self.yGate.A[i] = KA_IB.tau_h[i]
            self.yGate.B[i] = KA_IB.h_inf[i]
        self.xGate.tweakTau()
	self.yGate.tweakTau()


class K2(KChannel):
    v = ChannelBase.v_array
    m_inf = 1.0 / (1 + exp((-v - 10e-3) / 17e-3))
    tau_m = 1e-3 * (4.95 + 0.5 / (exp((v - 81e-3) / 25.6e-3) + \
                                      exp((-v - 132e-3) / 18e-3)))
    
    h_inf = 1.0 / (1 + exp((v + 58e-3) / 10.6e-3))
    tau_h = 1e-3 * (60 + 0.5 / (exp((v - 1.33e-3) / 200e-3) + \
					exp((-v - 130e-3) / 7.1e-3)))

    def __init__(self, name, parent, Ek=-95e-3):
        if config.context.exists(parent.path + '/' + name):
            KChannel.__init__(self, name, parent, 4, 1)
            return
	KChannel.__init__(self, name, parent, xpower=1.0, ypower=1.0, Ek=Ek)
	for i in range(config.ndivs + 1):
            self.xGate.A[i] = K2.tau_m[i]
            self.xGate.B[i] = K2.m_inf[i]
            self.yGate.A[i] = K2.tau_h[i]
            self.yGate.B[i] = K2.h_inf[i]
        self.xGate.tweakTau()
	self.yGate.tweakTau()
        self.X = 0.0
	# self.xGate.A.dumpFile('k2_xa.plot')
        # self.xGate.B.dumpFile('k2_xb.plot')
	# self.yGate.A.dumpFile('k2_ya.plot')
        # self.yGate.B.dumpFile('k2_yb.plot')
	

class KM(KChannel):
    v = ChannelBase.v_array
    a =  1e3 * 0.02 / ( 1 + exp((-v - 20e-3 ) / 5e-3))
    b = 1e3 * 0.01 * exp((-v - 43e-3) / 18e-3)
    minf = a / (a + b)
    mtau = 1 / (a + b)
    def __init__(self, name, parent, Ek=-95e-3):
        if config.context.exists(parent.path + '/' + name):
            KChannel.__init__(self, name, parent, 4, 1)
            return
	KChannel.__init__(self, name, parent, 1, Ek=Ek)
	for i in range(config.ndivs + 1):
            self.xGate.A[i] = KM.a[i]
            self.xGate.B[i] = KM.b[i]
	self.xGate.tweakAlpha()
        self.X = 0.0
        
class KCaChannel(KChannel):
    """[Ca+2] dependent K+ channel base class."""
    ca_min = 0.0
    ca_max = 1000.0
    ca_divs = 1000
    ca_conc = linspace(ca_min, ca_max, ca_divs + 1)

    def __init__(self, name, parent, xpower=0.0, ypower=0.0, zpower=1.0, Ek=-95e-3):
        if config.context.exists(parent.path + '/' + name):
            KChannel.__init__(self, name, parent, xpower, ypower, Ek=Ek)
            return
        KChannel.__init__(self, name, parent, xpower, ypower, Ek=Ek)
        self.connected_to_ca = False
        self.Zpower = zpower
        # self.zGate = moose.HHGate('zGate', self)
        self.zGate.A.xmin = KCaChannel.ca_min
        self.zGate.A.xdivs = KCaChannel.ca_divs
        self.zGate.A.xmax = KCaChannel.ca_max
        self.zGate.B.xmin = KCaChannel.ca_min
        self.zGate.B.xdivs = KCaChannel.ca_divs        
        self.zGate.B.xmax = KCaChannel.ca_max
        self.zGate.A.calcMode = 1
        self.zGate.B.calcMode = 1
        self.useConcentration = True
        self.addField('addmsg1')
        self.setField('addmsg1', '../CaPool . CONCEN Ca')
        config.LOGGER.debug('%s.addmsg1: %s' % (self.path,  self.getField('addmsg1')))
        for handler in config.LOGGER.handlers:
            handler.flush()


class KAHPBase(KCaChannel):
    def __init__(self, name, parent, Ek=-95e-3):
        if config.context.exists(parent.path + '/' + name):
            KCaChannel.__init__(self, name, parent)
            return
        KCaChannel.__init__(self, name, parent)
        self.Z = 0.0
        

class KAHP(KAHPBase):
    """AHP type K+ current"""

    alpha = where(KCaChannel.ca_conc < 100.0, 0.1 * KCaChannel.ca_conc, 10.0)
    beta =  ones(KCaChannel.ca_divs + 1) * 10.0

    def __init__(self, name, parent, xpower=0.0, ypower=0.0, zpower=1.0, Ek=-95e-3):
        if config.context.exists(parent.path + '/' + name):
            KAHPBase.__init__(self, name, parent, Ek=Ek)
            return
        KAHPBase.__init__(self, name, parent, Ek=Ek)
        for i in range(len(KAHP.alpha)):
            self.zGate.A[i] = KAHP.alpha[i]
            self.zGate.B[i] = KAHP.beta[i]
        self.zGate.tweakAlpha()
#         self.zGate.A.calcMode = 1
#         self.zGate.B.calcMode = 1


class KAHP_SLOWER(KAHPBase):

    alpha = where(KCaChannel.ca_conc < 500.0, 1e3 * KCaChannel.ca_conc / 50000, 10.0)
    beta =  ones(KCaChannel.ca_divs + 1) * 1.0

    def __init__(self, name, parent, Ek=-95e-3):
        if config.context.exists(parent.path + '/' + name):
            KAHPBase.__init__(self, name, parent, Ek=Ek)
            return
        KAHPBase.__init__(self, name, parent, Ek=Ek)
        for i in range(KCaChannel.ca_divs + 1):
            self.zGate.A[i] = KAHP_SLOWER.alpha[i]
            self.zGate.B[i] = KAHP_SLOWER.beta[i]
        self.zGate.tweakAlpha()
        self.zGate.A.calcMode = 1
        self.zGate.B.calcMode = 1


class KAHP_DP(KAHPBase):
    """KAHP for deep pyramidal cell"""
    alpha = where(KCaChannel.ca_conc < 100.0, 1e-1 * KCaChannel.ca_conc, 10.0)
    beta =  ones(KCaChannel.ca_divs + 1)
    def __init__(self, name, parent, Ek=-95e-3):
        if config.context.exists(parent.path + '/' + name):
            KAHPBase.__init__(self, name, parent, Ek=Ek)
            return
        KAHPBase.__init__(self, name, parent, Ek=Ek)
        for i in range(KCaChannel.ca_divs + 1):
            self.zGate.A[i] = KAHP_DP.alpha[i]
            self.zGate.B[i] = KAHP_DP.beta[i]
        self.zGate.tweakAlpha()

class KC(KCaChannel):
    """C type K+ channel"""
    alpha_ca = where(KCaChannel.ca_conc < 250.0, KCaChannel.ca_conc / 250.0, 1.0)
    v = ChannelBase.v_array
    alpha = where(v < -10e-3, 
                      2e3 / 37.95 * ( exp( ( v * 1e3 + 50 ) / 11 - ( v * 1e3 + 53.5 ) / 27 ) ),
                      2e3 * exp(( - v * 1e3 - 53.5) / 27))
    beta = where( v < -10e-3,
                  2e3 * exp(( - v * 1e3 - 53.5) / 27) - alpha, 
                  0.0)

    def __init__(self, name, parent, Ek=-95e-3):
        if config.context.exists(parent.path + '/' + name):
            KCaChannel.__init__(self, name, parent, Ek=Ek)
            return
        KCaChannel.__init__(self, name, parent, xpower=1.0, ypower=0.0, zpower=1.0, Ek=Ek)
        for i in range(KCaChannel.ca_divs + 1):
            self.zGate.A[i] = KC.alpha_ca[i]
            self.zGate.B[i] = 1.0
        for i in range(config.ndivs + 1):
            self.xGate.A[i] = KC.alpha[i] 
            self.xGate.B[i] = KC.beta[i]
        self.xGate.tweakAlpha()
        self.instant = 4
        self.X = 0.0
        
class KC_FAST(KC):
    """Fast KC channel"""
    def __init__(self, name, parent, Ek=-95e-3):
        if config.context.exists(parent.path + '/' + name):
            KC.__init__(self, name, parent, Ek=Ek)
            return
        KC.__init__(self, name, parent, Ek=Ek)
        for i in range(config.ndivs + 1):
            self.xGate.A[i] = 2 * self.xGate.A[i]
            self.xGate.B[i] = 2 * self.xGate.B[i]
        
if __name__ == "__main__":
#    a = KC_FAST('kc', moose.Neutral('/'))
#     b = KDR('KDR', moose.Neutral('/'))
    c = K2('K2', moose.Neutral('/'))

# 
# kchans.py ends here
