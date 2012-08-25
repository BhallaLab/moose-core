# savestate.py --- 
# 
# Filename: savestate.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat Aug 25 11:31:05 2012 (+0530)
# Version: 
# Last-Updated: Sat Aug 25 11:56:39 2012 (+0530)
#           By: subha
#     Update #: 52
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

def create_tree(tree, parentpath=''):
    if not tree:
        return
    node = tree[0]
    parent = moose.ematrix(parentpath + '/' + node[0], 1, node[1])
    if len(tree) < 2:
        return
    for sub in tree[1:]:
        create_tree(sub, parent.path)

if __name__ == '__main__':
    create_tree(test_tree)
    mu.printtree('/', ignorepaths=[])
    hu.savestate('test_moosestate.h5')


# 
# savestate.py ends here
