# channel.py --- 
# 
# Filename: channel.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Apr 17 15:17:35 2009 (+0530)
# Version: 
# Last-Updated: Sun May  3 23:37:38 2009 (+0530)
#           By: subhasis ray
#     Update #: 23
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

import numpy
import config
import moose

class ChannelBase(moose.HHChannel):
    v_array = numpy.linspace(config.vmin, config.vmax, config.ndivs + 1)
    def __init__(self, name, parent, xpower=1, ypower=0):
        moose.HHChannel.__init__(self, name, parent)
        if xpower != 0:
            self.Xpower = float(xpower)
            self.xGate = moose.HHGate(self.path + '/xGate')
            self.xGate.A.xmin = config.vmin
            self.xGate.A.xmax = config.vmax
            self.xGate.A.xdivs = config.ndivs
            self.xGate.B.xmin = config.vmin
            self.xGate.B.xmax = config.vmax
            self.xGate.B.xdivs = config.ndivs

        if ypower != 0:
            self.Ypower = float(ypower)
            self.yGate = moose.HHGate(self.path + '/yGate')
            self.yGate.A.xmin = config.vmin
            self.yGate.A.xmax = config.vmax
            self.yGate.A.xdivs = config.ndivs
            self.yGate.B.xmin = config.vmin
            self.yGate.B.xmax = config.vmax
            self.yGate.B.xdivs = config.ndivs
            
# 
# channel.py ends here
