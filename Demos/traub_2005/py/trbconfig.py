# trbconfig.py --- 
# 
# Filename: trbconfig.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Fri Mar  9 23:26:30 2012 (+0530)
# Version: 
# Last-Updated: Fri Mar  9 23:49:59 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 49
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
# 2012-03-09 23:26:34 (+0530) Subha started rewriting the code in
# config.py in old moose.
# 

# Code:

# List containing the name of the available channel classes.
from nachans import *
from kchans import *
from cachans import *
from archan import *
from capool import *

class TraubConfig(object):
    """Configuration class. There should be no instances of this
    class. Only a wrapper for methods."""
    channel_names = ['AR',
                     'CaPool',
                     'CaL',
                     'CaT',
                     'CaT_A',
                     'K2',
                     'KA',
                     'KA_IB',
                     'KAHP',
                     'KAHP_DP',
                     'KAHP_SLOWER',
                     'KC',
                     'KC_FAST',
                     'KDR',
                     'KDR_FS',
                     'KM',
                     'NaF',
                     'NaF2',
                     'NaF_TCR',
                     'NaP',
                     'NaPF',
                     'NaPF_SS',
                     'NaPF_TCR',
                     'NaF2_nRT']
    _channel_lib = {}
    @classmethod
    def init_channel_lib(cls):
        """Initialize the prototype channel library"""
        if not cls._channel_lib:
            if not moose.exists('/lib'):
                lib = moose.Neutral('/lib')
            for channel_name in cls.channel_names:
                cls._channel_lib[channel_name] = eval('%s("%s", "/lib")' % (channel_name, channel_name))
        return cls._channel_lib


if '__main__' == __name__:
    print TraubConfig.init_channel_lib()

# 
# trbconfig.py ends here
