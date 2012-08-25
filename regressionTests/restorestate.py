# restorestate.py --- 
# 
# Filename: restorestate.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat Aug 25 11:59:28 2012 (+0530)
# Version: 
# Last-Updated: Sat Aug 25 12:02:40 2012 (+0530)
#           By: subha
#     Update #: 11
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
import os
os.environ['NUMPTHREADS'] = '1'
sys.path.append('../python')
import moose
from moose import hdfutil as hu
from moose import utils as mu

test_tree = [('', 'Neutral'),
             [('a', 'Neutral'),
              [('c', 'Compartment'),
               [('d', 'SpikeGen'),],
               [('e', 'PulseGen'),]]],
             [('b', 'Neutral'), 
              [('f', 'Pool'), 
               [('g', 'Enz'), 
                [('h', 'Pool'),]], 
               [('i', 'Compartment'), ], 
               [('j', 'Nernst')]]]]

def check_tree(tree, parentpath=''):
    if not tree:
        return
    node = tree[0]
    parent = moose.to_el(parentpath + '/' + node[0], 1, node[1])
    if len(tree) < 2:
        return
    for sub in tree[1:]:
        check_tree(sub, parentpath + '/' + node[0])

if __name__ == '__main__':
    hu.restorestate('test_moosestate.h5')
    mu.printtree('/')
    check_tree(test_tree)
    


# 
# restorestate.py ends here
