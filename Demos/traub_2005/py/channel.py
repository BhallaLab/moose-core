# channel.py --- 
# 
# Filename: channel.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Apr 17 15:17:35 2009 (+0530)
# Version: 
# Last-Updated: Wed Jan  4 16:31:30 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 45
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
    def __init__(self, name, parent, xpower=1, ypower=0, Ek=0.0):
        path = None
        if isinstance(parent, str):
            path = '%s/%s' % (parent, name)
        elif isinstance(parent, moose.Neutral):
            path = '%s/%s' % (parent.path, name)
        elif isinstance(parent, moose.ObjId):
            path = '%s/%s' % (parent.getField('path'), name)
        print 'Creating', path
        if moose.exists(path):
            moose.HHChannel.__init__(self, path)
            return
        moose.HHChannel.__init__(self, path)            
        self.Ek = Ek
        if xpower != 0:
            self.Xpower = float(xpower)
            self.xGate = moose.HHGate(self.path + '/gateX')
            self.xGate.min = config.vmin
            self.xGate.max = config.vmax
            self.xGate.divs = config.ndivs

        if ypower != 0:
            self.Ypower = float(ypower)
            self.yGate = moose.HHGate(self.path + '/gateY')
            self.yGate.min = config.vmin
            self.yGate.max = config.vmax
            self.yGate.divs = config.ndivs
            
# 
# channel.py ends here
