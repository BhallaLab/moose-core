# mload.py --- 
# 
# Filename: mload.py
# Description: 
# Author: 
# Maintainer: 
# Created: Fri Feb  8 09:38:40 2013 (+0530)
# Version: 
# Last-Updated: Fri Feb  8 15:15:06 2013 (+0530)
#           By: subha
#     Update #: 162
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Utility to load models and detect filetype.
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
from moose import mtypes, neuroml


def loadFile(filename, target):
    """Try to load a model from specified `filename` under the element
    `target`.

    Returns
    -------
    a dict containing at least these three entries:
    
    modeltype: type of the loaded model.

    subtype: subtype of the loaded model, None if no specific subtype

    modelroot: root element of the model, None if could not be located - as is the case with Python scripts
    """
    istext = True
    with open(filename, 'rb') as infile:
        istext = mtypes.istextfile(infile)
    if not istext:
        print 'Cannot handle any binary formats yet'
        return None
    modeltype = mtypes.getType(filename)
    subtype = mtypes.getSubtype(filename, modeltype)
    if modeltype == 'genesis':
        if subtype == 'kkit' or subtype == 'prototype':
            model = moose.loadModel(filename, target)
        else:
            print 'Only kkit and prototype files can be loaded.'
    elif modeltype == 'xml' and subtype == 'neuroml':
        model = neuroml.loadNeuroML_L123(filename)
    # TODO: check with Aditya how to specify the target for
    # neuroml reader
    return {'modeltype': modeltype, 
            'subtype': subtype, 
            'model': model}


# 
# mload.py ends here
