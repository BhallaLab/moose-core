# trbchan.py --- 
# 
# Filename: trbchan.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Fri May  4 14:55:52 2012 (+0530)
# Version: 
# Last-Updated: Fri May  4 15:04:21 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 20
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Base class for channels in Traub model.
# 
# 

# Change log:
# 
# 2012-05-04 14:55:56 (+0530) subha started porting code from
# channel.py in old moose version to dh_branch.
# 

# Code:

import numpy as np
import trbconfig as cfg
import moose

class ChannelBase(moose.HHChannel):
    v_array = np.linspace(cfg.vmin, cfg.vmax, cfg.ndivs+1)
    def __init__(self, path, xpower=1, ypower=0, Ek=0.0):
        if moose.exists(path):
            moose.HHChannel.__init__(path)
            return
        self.Ek = Ek
        if xpower != 0:
            self.Xpower = xpower
            self.xGate = moose.HHGate('%s/gateX' % (path))
            self.xGate.min = cfg.vmin
            self.xGate.max = cfg.vmax
            self.xGate.divs = cfg.ndivs
        if ypower != 0:
            self.Ypower = ypower
            self.yGate = moose.HHGate('%s/gateY' % (path))
            self.yGate.min = cfg.vmin
            self.yGate.max = cfg.vmax
            self.yGate.divs = cfg.ndivs


# 
# trbchan.py ends here
