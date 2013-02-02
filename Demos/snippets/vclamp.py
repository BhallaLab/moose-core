# vclamp.py --- 
# 
# Filename: vclamp.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat Feb  2 19:16:54 2013 (+0530)
# Version: 
# Last-Updated: Sat Feb  2 19:31:08 2013 (+0530)
#           By: subha
#     Update #: 16
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

import sys
sys.path.append('../../python')

import moose

def vclamp_demo():
    container = moose.Neutral('/vClampDemo')
    clamp = moose.VClamp('/vClampDemo/vclamp')
    comp = moose.Compartment('/vClampDemo/compt')
    moose.connect(clamp, 'currentOut', comp, 'injectMsg')
    moose.connect(comp, 'VmOut', clamp, 'voltageIn')
    moose.setClock(0, 0.01)
    moose.useClock(0, comp.path, 'process')
    moose.useClock(0, clamp.path, 'process')    
    moose.reinit()
    moose.start(1.0)

if __name__ == '__main__':
    vclamp_demo()
    

# 
# vclamp.py ends here
