# init.py --- 
# 
# Filename: init.py
# Description: 
# Author: 
# Maintainer: 
# Created: Wed May 23 21:24:11 2012 (+0530)
# Version: 
# Last-Updated: Thu May 24 16:54:32 2012 (+0530)
#           By: subha
#     Update #: 63
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Initialize model prototypes
# 
# 

# Change log:
# 
# 
# 
# 
# Code:

import config
import nachans 
import kchans 
import archan
import cachans
import capool


_channels = {}

def init_chanlib():
    """Return a dict of channel name, channel prototype pairs. If the
    channel prototypes have not been initialized, this functions
    initializes the same."""
    global _channels
    if _channels:
        return _channels
    _channels.update(nachans.initNaChannelPrototypes())
    _channels.update(kchans.initKChannelPrototypes())
    _channels.update(archan.initARChannelPrototypes())
    _channels.update(cachans.initCaChannelPrototypes())
    _channels.update(initCaPoolPrototype())    
    return _channels

        
        
# 
# init.py ends here
