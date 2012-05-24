# trbchan.py --- 
# 
# Filename: trbchan.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Fri May  4 14:55:52 2012 (+0530)
# Version: 
# Last-Updated: Thu May 24 15:02:11 2012 (+0530)
#           By: subha
#     Update #: 22
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
    vmin = -120e-3
    vmax = 40e-3
    ndivs = 640
    v_array = np.linspace(vmin, vmax, ndivs+1)
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
