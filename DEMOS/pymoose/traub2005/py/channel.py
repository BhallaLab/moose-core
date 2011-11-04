# channel.py --- 
# 
# Filename: channel.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Apr 17 15:17:35 2009 (+0530)
# Version: 
# Last-Updated: Fri Oct 21 15:58:15 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 24
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

import numpy
import config
import moose

class ChannelBase(moose.HHChannel):
    v_array = numpy.linspace(config.vmin, config.vmax, config.ndivs + 1)
    def __init__(self, name, parent, xpower=1, ypower=0):
        moose.HHChannel.__init__(self, name, parent)
        if xpower != 0:
            self.Xpower = float(xpower)
            # self.xGate = moose.HHGate(self.path + '/xGate')
            self.xGate.A.xmin = config.vmin
            self.xGate.A.xmax = config.vmax
            self.xGate.A.xdivs = config.ndivs
            self.xGate.B.xmin = config.vmin
            self.xGate.B.xmax = config.vmax
            self.xGate.B.xdivs = config.ndivs

        if ypower != 0:
            self.Ypower = float(ypower)
            # self.yGate = moose.HHGate(self.path + '/yGate')
            self.yGate.A.xmin = config.vmin
            self.yGate.A.xmax = config.vmax
            self.yGate.A.xdivs = config.ndivs
            self.yGate.B.xmin = config.vmin
            self.yGate.B.xmax = config.vmax
            self.yGate.B.xdivs = config.ndivs
            
# 
# channel.py ends here
