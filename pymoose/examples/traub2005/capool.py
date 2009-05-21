# CaPool.py --- 
# 
# Filename: capool.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Wed Apr 22 22:21:11 2009 (+0530)
# Version: 
# Last-Updated: Tue May  5 17:28:37 2009 (+0530)
#           By: subhasis ray
#     Update #: 57
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Implements the Ca2+ pool
# 
# 

# Change log:
# 
# 
# 
# 
 

# Code:

import moose
import config
from cachans import CaL
from kchans import KCaChannel

class CaPool(moose.CaConc):
    def __init__(self, *args):
	moose.CaConc.__init__(self, *args)
        self.CaBasal = 0.0        
        
    def connectCaChannels(self, channel_list):
        """Connects the Ca2+ channels in channel_list as a source of
        Ca2+ to the pool."""
        for channel in channel_list:
                if not hasattr(channel, 'connected_to_pool') or not channel.connected_to_pool:
                    channel.connect('IkSrc', self, 'current')
                    channel.connected_to_pool = True
                else:
                    print channel.path, 'already connected'
                
    def connectDepChannels(self, channel_list):
        """Connect channels in channel_list as dependent channels"""
        for channel in channel_list:
            if channel.useConcentration == 0:
                print "WRANING: This channel does not use concentration:", channel.path
            elif not hasattr(channel, 'connected_to_ca') or not channel.connected_to_ca:
                self.connect("concSrc", channel, "concen")
                channel.connected_to_ca = True
            else:
                print "WARNING: Ignoring non-KCaChannel", channel.path
# 
# capool.py ends here
