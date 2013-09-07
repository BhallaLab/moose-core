# units.py --- 
# 
# Filename: units.py
# Description: 
# Author: 
# Maintainer: 
# Created: Thu Jul 25 16:30:14 2013 (+0530)
# Version: 
# Last-Updated: Thu Jul 25 17:55:16 2013 (+0530)
#           By: subha
#     Update #: 50
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

"""Utility to create mapping from NeuroML2 unit strings (as specified
in NeuroMLCoreDimensions.xml) to SI

"""
import numpy as np
import debug.debug as debug
import re
try:
    from lxml import etree
    debug.printDebug("DEBUG", "running with lxml.etree")
except ImportError:
    try:
        # Python 2.5
        import xml.etree.cElementTree as etree
        debug.printDebug("DEBUG", "running with cElementTree")
    except ImportError:
        try:
            # Python 2.5
            import xml.etree.ElementTree as etree
            debug.printDebug("DEBUG", "running with ElementTree")
        except ImportError:
            try:
              # normal cElementTree install
              import cElementTree as etree
              debug.printDebug("DEBUG", "running with cElementTree")
            except ImportError:
                try:
                    # normal ElementTree install
                    import elementtree.ElementTree as etree
                    debug.printDebug("DEBUG", "running with ElementTree")
                except ImportError:
                    debug.prefix("FATAL", "Failed to import ElementTree")
                    os._exit(1)



unitXml = 'xml/NeuroMLCoreDimensions.xml'

import os 
import sys
if os.path.exists(unitXml) :
    unitsdoc = etree.parse(unitXml)
else:
    debug.printDebug("FATAL", "File {0} not found.".format(unitXml))
    sys.exit(0)
    

dims = dict([(el.name, el) for el in unitsdoc.getroot() if el.tag == 'Dimension'])
units = dict([(el.symbol, el) for el in unitsdoc.getroot() if el.tag == 'Unit'])

# optional +/- followed by 0 or more digits followed by optional
# decimal point followed by optional digits followed by optional
# (exponent symbol followed by optional +/- followed by one or more
# digits.
magnitude_regex = re.compile(r'^[\+-]?\d*\.?\d*([eE][\+-]?\d+)?')

def SI(expression):
    match = magnitude_regex.match(expression)
    magnitude = float(match.group(0))
    unitstr = re.split(magnitude_regex, expression)[-1]
    try:
        unit = units[unitstr]
    except KeyError:
        return magnitude
    try:
        return magnitude * np.power(10, int(unit.power))
    except AttributeError: # degC has offset in stead of magnitude
        return magnitude + float(unit.offset)
    
# 
# units.py ends here
