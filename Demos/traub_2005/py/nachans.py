# nachans.py --- 
# 
# Filename: nachans.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Apr 17 23:58:13 2009 (+0530)
# Version: 
# Last-Updated: Thu May 24 09:32:59 2012 (+0530)
#           By: subha
#     Update #: 217
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

# Code:

from numpy import where, linspace, exp
#import pylab
import moose

import config
from channel import ChannelBase

class NaChannel(ChannelBase):
    """Dummy base class for all Na+ channels"""
    def __init__(self, path, xpower, ypower=0.0, Ek=50e-3):
        print 'NaChannel.__init__:', name, parent.path
        ChannelBase.__init__(self, path, xpower=xpower, ypower=ypower, Ek=Ek)

class NaF(NaChannel):
    def __init__(self, path, shift=-3.5e-3, Ek=50e-3):
        if moose.exists(parent.path + '/' + name):
            NaChannel.__init__(self, path, xpower=3.0, ypower=1.0, Ek=Ek)
            return
        NaChannel.__init__(self, path, xpower=3.0, ypower=1.0, Ek=Ek)
        v = linspace(config.vmin, config.vmax, config.ndivs + 1) + shift
        tau_m = where(v < -30e-3, \
                          1.0e-3 * (0.025 + 0.14 * exp((v + 30.0e-3) / 10.0e-3)), \
                          1.0e-3 * (0.02 + 0.145 * exp(( - v - 30.0e-3) / 10.0e-3)))
        m_inf = 1.0 / (1.0 + exp(( - v - 38e-3) / 10e-3))
        v = v - shift
        tau_h = 1.0e-3 * (0.15 + 1.15 / ( 1.0 + exp(( v + 37.0e-3) / 15.0e-3)))
        h_inf = 1.0 / (1.0 + exp((v + 62.9e-3) / 10.7e-3))
        self.xGate.tableA = tau_m
        self.xGate.tableB = m_inf
        self.yGate.tableA = tau_h
        self.yGate.tableB = h_inf
        self.xGate.tweakTau()
        self.yGate.tweakTau()
        self.X = 0.0
        
class NaF2(NaChannel):
    def __init__(self, path, shift=-2.5e-3, Ek=50e-3):
        if moose.exists(parent.path + '/' + name):
            NaChannel.__init__(self, path, xpower=3.0, ypower=1.0, Ek=Ek)
            return
        NaChannel.__init__(self, path, xpower=3.0, ypower=1.0, Ek=Ek)
        config.LOGGER.debug('NaF2: shift = %g' % (shift))
        v = linspace(config.vmin, config.vmax, config.ndivs + 1)
        tau_h = 1e-3 * (0.225 + 1.125 / ( 1 + exp( (  v + 37e-3 ) / 15e-3 ) ))
        
        h_inf = 1.0 / (1.0 + exp((v + 58.3e-3) / 6.7e-3))
        v = v + shift
        tau_m = where(v < -30e-3, \
                          1.0e-3 * (0.0125 + 0.1525 * exp ((v + 30e-3) / 10e-3)), \
                          1.0e-3 * (0.02 + 0.145 * exp((-v - 30e-3) / 10e-3)))
        
        m_inf = 1.0 / (1.0 + exp(( - v - 38e-3) / 10e-3))

        for i in range(config.ndivs + 1):
            self.xGate.A[i] = tau_m[i]
            self.xGate.B[i] = m_inf[i]
            self.yGate.A[i] = tau_h[i]
            self.yGate.B[i] = h_inf[i]
        self.xGate.tweakTau()
        self.yGate.tweakTau()
        self.X = 0.0

class NaF2_nRT(NaF2):
    """This is a version of NaF2 without the fastNa_shift - applicable to nRT cell."""
    def __init__(self, path):
        NaF2.__init__(self, path, shift=0.0)


class NaP(NaChannel):
    def __init__(self, path, Ek=50e-3):
        if moose.exists(parent.path + '/' + name):
            NaChannel.__init__(self, path, xpower=1.0, Ek=Ek)
            return
        NaChannel.__init__(self, path, xpower=1.0, Ek=Ek)
        v = linspace(config.vmin, config.vmax, config.ndivs + 1)
        tau_m = where(v < -40e-3, \
                          1.0e-3 * (0.025 + 0.14 * exp((v + 40e-3) / 10e-3)), \
                          1.0e-3 * (0.02 + 0.145 * exp((-v - 40e-3) / 10e-3)))
        m_inf = 1.0 / (1.0 + exp((-v - 48e-3) / 10e-3))
        for i in range(config.ndivs + 1):
            self.xGate.A[i] = tau_m[i]
            self.xGate.B[i] = m_inf[i]
        self.xGate.tweakTau()
        self.X = 0.0


class NaPF(NaChannel):
    """Persistent Na+ current, fast"""
    def __init__(self, path, Ek=50e-3):
        if moose.exists(parent.path + '/' + name):
            NaChannel.__init__(self, path, xpower=3.0, Ek=Ek)
            return
        NaChannel.__init__(self, path, xpower=3.0, Ek=Ek)
        v = linspace(config.vmin, config.vmax, config.ndivs + 1)
        tau_m = where(v < -30e-3, \
                           1.0e-3 * (0.025 + 0.14 * exp((v  + 30.0e-3) / 10.0e-3)), \
                           1.0e-3 * (0.02 + 0.145 * exp((- v - 30.0e-3) / 10.0e-3)))
        m_inf = 1.0 / (1.0 + exp((-v - 38e-3) / 10e-3))
        for i in range(config.ndivs + 1):
            self.xGate.A[i] = tau_m[i]
            self.xGate.B[i] = m_inf[i]
        self.xGate.tweakTau()


class NaPF_SS(NaChannel):
    def __init__(self, path, shift=-2.5e-3, Ek=50e-3):
        if moose.exists(parent.path + '/' + name):
            NaChannel.__init__(self, path, xpower=3.0, Ek=Ek)
            return
        NaChannel.__init__(self, path, xpower=3.0, Ek=Ek)
        config.LOGGER.debug('NaPF_SS: shift = %g' % (shift))
        v = linspace(config.vmin, config.vmax, config.ndivs + 1) + shift
        tau_m = where(v < -30e-3, \
                           1.0e-3 * (0.025 + 0.14 * exp((v  + 30.0e-3) / 10.0e-3)), \
                           1.0e-3 * (0.02 + 0.145 * exp((- v - 30.0e-3) / 10.0e-3)))
        m_inf = 1.0 / (1.0 + exp((- v - 38e-3) / 10e-3))
        for i in range(config.ndivs + 1):
            self.xGate.A[i] = tau_m[i]
            self.xGate.B[i] = m_inf[i]
        self.xGate.tweakTau()


class NaPF_TCR(NaChannel):
    """Persistent Na+ channel specific to TCR cells. Only difference
    with NaPF is power of m is 1 as opposed 3."""
    def __init__(self, path, shift=7e-3, Ek=50e-3):
        if moose.exists(parent.path + '/' + name):
            NaChannel.__init__(self, path, xpower=1.0, Ek=Ek)
            return 
        NaChannel.__init__(self, path, xpower=1.0, Ek=Ek)
        v = linspace(config.vmin, config.vmax, config.ndivs + 1) + shift
        tau_m = where(v < -30e-3, \
                           1.0e-3 * (0.025 + 0.14 * exp((v  + 30.0e-3) / 10.0e-3)), \
                           1.0e-3 * (0.02 + 0.145 * exp((- v - 30.0e-3) / 10.0e-3)))
        m_inf = 1.0 / (1.0 + exp((-v - 38e-3) / 10e-3))
        for i in range(config.ndivs + 1):
            self.xGate.A[i] = tau_m[i]
            self.xGate.B[i] = m_inf[i]
        self.xGate.tweakTau()

class NaF_TCR(NaChannel):
    """Fast Na+ channel for TCR cells. This is almost identical to
    NaF, but there is a nasty voltage shift in the tables."""
    def __init__(self, path, Ek=50e-3):
        if moose.exists(parent.path + '/' + name):
            NaChannel.__init__(self, path, xpower=3.0, ypower=1.0, Ek=Ek)
            return
        NaChannel.__init__(self, path, xpower=3.0, ypower=1.0, Ek=Ek)
        shift_mnaf = -5.5e-3
        shift_hnaf = -7e-3
        v = linspace(config.vmin, config.vmax, config.ndivs + 1) 
        tau_h = 1.0e-3 * (0.15 + 1.15 / ( 1.0 + exp(( v + 37.0e-3) / 15.0e-3)))        
        h_inf = 1.0 / (1.0 + exp((v + shift_hnaf + 62.9e-3) / 10.7e-3))
        v = v + shift_mnaf
        tau_m = where(v < -30e-3, \
                          1.0e-3 * (0.025 + 0.14 * exp((v + 30.0e-3) / 10.0e-3)), \
                          1.0e-3 * (0.02 + 0.145 * exp(( - v - 30.0e-3) / 10.0e-3)))
        m_inf = 1.0 / (1.0 + exp(( - v - 38e-3) / 10e-3))

        for i in range(config.ndivs + 1):
            self.xGate.A[i] = tau_m[i]
            self.xGate.B[i] = m_inf[i]
            self.yGate.A[i] = tau_h[i]
            self.yGate.B[i] = h_inf[i]
        self.xGate.tweakTau()
        self.yGate.tweakTau()

def initNaChannelPrototypes(libpath='/library'):
    channel_names = [
        'NaF',
        'NaF2',
        'NaF2_nRT',
        'NaP',
        'NaPF',
        'NaPF_SS',
        'NaPF_TCR',
        'NaF_TCR',
        ]
    prototypes = {}
    for channel_name in channels_names:
        channel_class = eval(channel_name)
        prototypes[channel_name] = channel_class(libpath, channel_name)        
    return prototypes

# 
# nachans.py ends here
