# kchans.py --- 
# 
# Filename: kchans.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Apr 17 23:58:49 2009 (+0530)
# Version: 
# Last-Updated: Sat May 26 14:42:23 2012 (+0530)
#           By: subha
#     Update #: 855
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
from channelbase import ChannelBase
from numpy import where, linspace, exp, arange, ones, zeros, array
import numpy as np


class KChannel(ChannelBase):
    """This is a dummy base class to keep type information."""
    _prototypes = {}
    def __init__(self, path, xpower=1, ypower=0, Ek=-95e-3):
        if moose.exists(path):
            ChannelBase.__init__(self, path, xpower, ypower)
            return
        ChannelBase.__init__(self, path, xpower, ypower)
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
    
    def __init__(self, path, xpower=4, ypower=0, Ek=-95e-3):
        if moose.exists(path):
            KChannel.__init__(self, path)
            return
        KChannel.__init__(self, path, xpower, ypower, Ek=Ek)
        self.xGate.tableA = KDR.m_inf / KDR.tau_m
        self.xGate.tableB = 1 / KDR.tau_m
        self.X = 0.0


class KDR_FS(KChannel):
    """KDR for fast spiking neurons"""
    v = ChannelBase.v_array
    m_inf = 1.0 / (1.0 + exp((- v - 27e-3) / 11.5e-3))
    tau_m =  where(v < -10e-3, \
                       1e-3 * (0.25 + 4.35 * exp((v + 10.0e-3) / 10.0e-3)), \
                       1e-3 * (0.25 + 4.35 * exp((- v - 10.0e-3) / 10.0e-3)))

    def __init__(self, path, Ek=-95e-3):
        if moose.exists(path):
            KChannel.__init__(self, path, xpower=4, ypower=0, Ek=Ek)
            return
        KChannel.__init__(self, path, 4, Ek=Ek)
        self.xGate.tableA = KDR_FS.m_inf / KDR_FS.tau_m
        self.xGate.tableB = 1 / KDR_FS.tau_m
        self.X = 0.0


class KA(KChannel):
    """A type K+ channel"""
    v = ChannelBase.v_array
    m_inf = 1 / ( 1 + exp( ( - v - 60e-3 ) / 8.5e-3 ) )
    tau_m =  1e-3 * (0.185 + 0.5 / ( exp( ( v + 35.8e-3 ) / 19.7e-3 ) + exp( ( - v - 79.7e-3 ) / 12.7e-3 ) ))
    h_inf =   1 / ( 1 + exp( ( v + 78e-3 ) / 6e-3 ) )
    tau_h = where( v <= -63e-3,\
                       1e-3 * 0.5 / ( exp( ( v + 46e-3 ) / 5e-3 ) + exp( ( - v - 238e-3 ) / 37.5e-3 ) ), \
                       9.5e-3)

    def __init__(self, path, Ek=-95e-3):
        if moose.exists(path):
            KChannel.__init__(self, path, 4, 1)
            return
        KChannel.__init__(self, path, 4, 1)
        self.xGate.tableA = KA.m_inf / KA.tau_m
        self.xGate.tableB = 1  / KA.tau_m
        self.yGate.tableA = KA.h_inf / KA.tau_h
        self.yGate.tableB = 1 / KA.tau_h
        self.X = 0.0

class KA_IB(KChannel):
    """A type K+ channel for tufted intrinsically bursting cells -
    multiplies tau_h of KA by 2.6"""
    # v = ChannelBase.v_array
    # m_inf = 1 / ( 1 + exp( ( - v - 60e-3 ) / 8.5e-3 ) )
    # tau_m =  1e-3 * (0.185 + 0.5 / ( exp( ( v + 35.8e-3 ) / 19.7e-3 ) + exp( ( - v - 79.7e-3 ) / 12.7e-3 ) ))
    # h_inf =   1 / ( 1 + exp( ( v + 78e-3 ) / 6e-3 ) )
    # tau_h = 2.6 * where( v <= -63e-3,\
    #                          1e-3 * 0.5 / ( exp( ( v + 46e-3 ) / 5e-3 ) + exp( ( - v - 238e-3 ) / 37.5e-3 ) ), \
    #                          9.5e-3)

    def __init__(self, path, Ek=-95e-3):
        if moose.exists(path):
            KChannel.__init__(self, path, 4, 1)
            return
        KChannel.__init__(self, path, 4, 1, Ek=Ek)
        self.xGate.tableA = KA.m_inf / KA.tau_m
        self.xGate.tableB = 1 / KA.tau_m 
        self.yGate.tableA = KA.h_inf / (2.6*KA.tau_h)
        self.yGate.tableB = 1 / (2.6*KA.tau_h)
        self.X = 0.0


class K2(KChannel):
    v = ChannelBase.v_array
    m_inf = 1.0 / (1 + exp((-v - 10e-3) / 17e-3))
    tau_m = 1e-3 * (4.95 + 0.5 / (exp((v - 81e-3) / 25.6e-3) + \
                                      exp((-v - 132e-3) / 18e-3)))
    
    h_inf = 1.0 / (1 + exp((v + 58e-3) / 10.6e-3))
    tau_h = 1e-3 * (60 + 0.5 / (exp((v - 1.33e-3) / 200e-3) + \
					exp((-v - 130e-3) / 7.1e-3)))

    def __init__(self, path, Ek=-95e-3):
        if moose.exists(path):
            KChannel.__init__(self, path, xpower=1.0, ypower=1.0)
            return
        KChannel.__init__(self, path, xpower=1.0, ypower=1.0, Ek=Ek)
        self.xGate.tableA = K2.m_inf / K2.tau_m
        self.xGate.tableB = 1 / K2.tau_m
        self.yGate.tableA = K2.h_inf / K2.tau_h
        self.yGate.tableB = 1 / K2.tau_h
        self.X = 0.0
	

class KM(KChannel):
    v = ChannelBase.v_array
    a =  1e3 * 0.02 / ( 1 + exp((-v - 20e-3 ) / 5e-3))
    b = 1e3 * 0.01 * exp((-v - 43e-3) / 18e-3)
    def __init__(self, path, Ek=-95e-3):
        if moose.exists(path):
            KChannel.__init__(self, path, xpower=1.0, ypower=0.0)
            return
        KChannel.__init__(self, path, xpower=1.0, ypower=0.0, Ek=Ek)
        self.xGate.tableA = KM.a
        self.xGate.tableB = KM.b + KM.b
        self.X = 0.0

        
class KCaChannel(KChannel):
    """[Ca+2] dependent K+ channel base class."""
    ca_min = 0.0
    ca_max = 1000.0
    ca_divs = 1000
    ca_conc = linspace(ca_min, ca_max, ca_divs + 1)

    def __init__(self, path, xpower=0.0, ypower=0.0, zpower=1.0, Ek=-95e-3):
        if moose.exists(path):
            KChannel.__init__(self, path, xpower, ypower, Ek=Ek)
            return
        KChannel.__init__(self, path, xpower, ypower, Ek=Ek)
        self.connected_to_ca = False
        self.Zpower = zpower
        self.zGate = moose.HHGate('%s/gateZ' % (self.path))
        self.zGate.min = KCaChannel.ca_min
        self.zGate.divs = KCaChannel.ca_divs
        self.zGate.max = KCaChannel.ca_max
        self.zGate.useInterpolation = False
        self.useConcentration = True
        ca_msg_field = moose.Mstring('%s/addmsg1' % (self.path))
        ca_msg_field.value = '../CaPool	concOut	. concen'


class KAHPBase(KCaChannel):
    def __init__(self, path, xpower=0.0, ypower=0.0, zpower=1.0, Ek=-95e-3):
        if moose.exists(path):
            KCaChannel.__init__(self, path, xpower=xpower, ypower=ypower, zpower=zpower, Ek=Ek)
            return
        KCaChannel.__init__(self, path, xpower=xpower, ypower=ypower, zpower=zpower, Ek=Ek)
        self.Z = 0.0
        

class KAHP(KAHPBase):
    """AHP type K+ current"""

    alpha = where(KCaChannel.ca_conc < 100.0, 0.1 * KCaChannel.ca_conc, 10.0)
    beta =  ones(KCaChannel.ca_divs + 1) * 10.0

    def __init__(self, path, xpower=0.0, ypower=0.0, zpower=1.0, Ek=-95e-3):
        if moose.exists(path):
            KAHPBase.__init__(self, path, xpower=xpower, ypower=ypower, zpower=zpower, Ek=Ek)
            return
        KAHPBase.__init__(self, path, xpower=xpower, ypower=ypower, zpower=zpower, Ek=Ek)
        self.zGate.tableA = KAHP.alpha
        self.zGate.tableB =  KAHP.alpha + KAHP.beta
#         self.zGate.A.calcMode = 1
#         self.zGate.B.calcMode = 1


class KAHP_SLOWER(KAHPBase):

    alpha = where(KCaChannel.ca_conc < 500.0, 1e3 * KCaChannel.ca_conc / 50000, 10.0)
    beta =  ones(KCaChannel.ca_divs + 1) * 1.0

    def __init__(self, path, Ek=-95e-3):
        if moose.exists(path):
            KAHPBase.__init__(self, path, Ek=Ek)
            return
        KAHPBase.__init__(self, path, Ek=Ek)
        self.zGate.tableA = KAHP_SLOWER.alpha
        self.zGate.tableB = KAHP_SLOWER.alpha + KAHP_SLOWER.beta
        # self.zGate.tableA.calcMode = 1
        # self.zGate.tableB.calcMode = 1


class KAHP_DP(KAHPBase):
    """KAHP for deep pyramidal cell"""
    alpha = where(KCaChannel.ca_conc < 100.0, 1e-1 * KCaChannel.ca_conc, 10.0)
    beta =  ones(KCaChannel.ca_divs + 1)
    def __init__(self, path, Ek=-95e-3):
        if moose.exists(path):
            KAHPBase.__init__(self, path, Ek=Ek)
            return
        KAHPBase.__init__(self, path, Ek=Ek)
        self.zGate.tableA = KAHP_DP.alpha
        self.zGate.tableB = KAHP_DP.alpha + KAHP_DP.beta

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

    def __init__(self, path, Ek=-95e-3):
        if moose.exists(path):
            KCaChannel.__init__(self, path, Ek=Ek)
            return
        KCaChannel.__init__(self, path, xpower=1.0, ypower=0.0, zpower=1.0, Ek=Ek)
        self.zGate.tableA = KC.alpha_ca
        self.zGate.tableB = ones(KCaChannel.ca_divs + 1)
        self.xGate.tableA = KC.alpha
        self.xGate.tableB = KC.alpha + KC.beta
        self.instant = 4
        self.X = 0.0

        
class KC_FAST(KC):
    """Fast KC channel"""
    def __init__(self, path, Ek=-95e-3):
        if moose.exists(path):
            KC.__init__(self, path, Ek=Ek)
            return
        KC.__init__(self, path, Ek=Ek)
        self.xGate.tableA = 2 * KC.alpha
        self.xGate.tableB = 2 * (KC.alpha + KC.beta)

        
def initKChannelPrototypes(libpath='/library'):
    if KChannel._prototypes:
        return KChannel._prototypes
    channel_names = ['KDR', 
                     'KDR_FS', 
                     'KA', 
                     'KA_IB',
                     'K2', 
                     'KM', 
                     'KAHP',
                     'KAHP_SLOWER',
                     'KAHP_DP',
                     'KC',
                     'KC_FAST']    
    for channel_name in channel_names:
        channel_class = eval(channel_name)
        path = '%s/%s' % (libpath, channel_name)
        KChannel._prototypes[channel_name] = channel_class(path)
        print 'Created channel prototype:', path
    return KChannel._prototypes
        

# 
# kchans.py ends here
