# CaPool.py --- 
# 
# Filename: capool.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Wed Apr 22 22:21:11 2009 (+0530)
# Version: 
# Last-Updated: Wed Sep 16 18:10:50 2009 (+0530)
#           By: subhasis ray
#     Update #: 58
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
import config
from cachans import CaL
from kchans import KCaChannel

class CaPool(moose.CaConc):
    def __init__(self, *args):
	moose.CaConc.__init__(self, *args)
        self.CaBasal = 0.0        
        config.context.runG('setfield ' + self.path + ' ceiling 1e6')
        config.context.runG('setfield ' + self.path + ' floor 0.0')
        
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
