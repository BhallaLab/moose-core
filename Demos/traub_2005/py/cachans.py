# cachans.py --- 
# 
# Filename: cachans.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Sat Apr 18 00:18:24 2009 (+0530)
# Version: 
# Last-Updated: Fri May 25 14:13:26 2012 (+0530)
#           By: subha
#     Update #: 245
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

from numpy import where, exp, array
import moose
from channelbase import ChannelBase

class CaChannel(ChannelBase):
    """This is just a place holder to maintain type information"""
    def __init__(self, path, xpower=1.0, ypower=0.0, Ek=125e-3):
        if moose.exists(path):
            ChannelBase.__init__(self, path, xpower=xpower, ypower=ypower)
            return
        ChannelBase.__init__(self, path, xpower=xpower, ypower=ypower)
        self.Ek = Ek

class CaL(CaChannel):
    """Low threshold calcium channel"""
    v = array(ChannelBase.v_array)
    alpha = 1.6e3 / (1.0 + exp(-0.072 * (v * 1e3 - 5)))
    v = v + 8.9e-3
    beta = where( abs(v) * 1e3 < 1e-6,
                  1e3 * 0.1 * exp(-v / 5e-3),
                  1e3 * 0.02 * v * 1e3 / (exp(v / 5e-3) - 1))

    def __init__(self, path):
        if moose.exists(path):
            CaChannel.__init__(self, path, xpower=2.0, Ek=125e-3)
            return
        CaChannel.__init__(self, path, xpower=2.0, Ek=125e-3)
        self.xGate.tableA = CaL.alpha
        self.xGate.tableB = CaL.beta
        self.xGate.tweakAlpha()
        self.X = 0.0
        ca_msg_field = moose.Mstring('%s/addmsg1' % (self.path))
        ca_msg_field.value = '.	IkOut	../CaPool	current'


class CaT(CaChannel):
    v = ChannelBase.v_array
    m_inf = 1 / (1 + exp( (- v - 56e-3) / 6.2e-3))
    tau_m = 1e-3 * (0.204 + 0.333 / ( exp(( v + 15.8e-3) / 18.2e-3 ) + 
                                      exp((- v - 131e-3) / 16.7e-3)))
    h_inf = 1 / (1 + exp(( v + 80e-3 ) / 4e-3))
    tau_h = where( v < -81e-3, 
                   1e-3 * 0.333 * exp( ( v + 466e-3 ) / 66.6e-3 ),
                   1e-3 * (9.32 + 0.333 * exp( ( -v - 21e-3 ) / 10.5e-3 )))

    def __init__(self, path):
        if moose.exists(path):
            CaChannel.__init__(self, path, xpower=2.0, ypower=1.0)
            return
        CaChannel.__init__(self, path, xpower=2.0, ypower=1.0)
        self.Ek = 125e-3
        self.X = 0.0
        self.xGate.tableA = CaT.tau_m
        self.xGate.tableB = CaT.m_inf
        self.yGate.tableA = CaT.tau_h
        self.yGate.tableB = CaT.h_inf
        self.xGate.tweakTau()
        self.yGate.tweakTau()


class CaT_A(CaChannel):
    v = ChannelBase.v_array
    m_inf  = 1.0 / ( 1 + exp( ( - v * 1e3 - 52 ) / 7.4 ) )
    tau_m  = 1e-3 * (1 + .33 / ( exp( ( v * 1e3 + 27.0 ) / 10.0 ) + exp( ( - v * 1e3 - 102 ) / 15.0 )))
    
    h_inf  = 1 / ( 1 + exp( ( v * 1e3 + 80 ) / 5 ) )
    tau_h = 1e-3 * (28.30 + 0.33 / (exp(( v * 1e3 + 48.0)/ 4.0) + exp( ( -v * 1e3 - 407.0) / 50.0 ) ))

    def __init__(self, path):
        if moose.exists(path):
            CaChannel.__init__(self, path, xpower=2.0, ypower=1.0, Ek=125e-3)
            return
        CaChannel.__init__(self, path, xpower=2.0, ypower=1.0, Ek=125e-3)
        self.Ek = 125e-3
        self.xGate.tableA = CaT_A.tau_m
        self.xGate.tableB = CaT_A.m_inf
        self.yGate.tableA = CaT_A.tau_h
        self.yGate.tableB = CaT_A.h_inf
        self.xGate.tweakTau()
        self.yGate.tweakTau()
        self.X = 0

def initCaChannelPrototypes(libpath='/library'):
    channel_names = ['CaL', 'CaT', 'CaT_A']
    prototypes = {}
    for channel_name in channel_names:
        channel_class = eval(channel_name)
        channel = channel_class('%s/%s' % (libpath, channel_name))
        prototypes[channel_name] = channel
    return prototypes

# 
# cachans.py ends here
