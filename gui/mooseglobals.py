# mooseglobals.py --- 
# 
# Filename: mooseglobals.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Wed Jan 27 16:45:38 2010 (+0530)
# Version: 
# Last-Updated: Fri Sep 21 12:43:09 2012 (+0530)
#           By: subha
#     Update #: 10
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Global constants and variables for moose gui.
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
from datetime import date

class MooseGlobals:
    # Text fragments with info on MOOSE
    TITLE_TEXT = 'Multiscale Object Oriented Simulation Environment (MOOSE)'
    COPYRIGHT_TEXT = 'Copyright (C) 2003-%s Upinder S. Bhalla and NCBS.'  % (str(date.today().year))
    LICENSE_TEXT = 'It is made available under the terms of the GNU Lesser General Public License version 2.1. See the file COPYING.LIB for the full notice.'
    ABOUT_TEXT = 'MOOSE is a simulation environment for Computational Biology. It provides a general messaging framework. It is currently focused on facilitating neuronal and chemical kinetics simulations.'
    WEBSITE = 'http://moose.ncbs.res.in'
    # These constants are for selectig a GUI mode.
    MODE_ADVANCED = 0 # Everything is open to the user
    MODE_KKIT = 1     # Should imitate kinetikit
    MODE_NKIT = 2     # Should imitate neurokit
    CMD_MODE_GENESIS = 0 # The command line is interpreted as GENESIS command
    CMD_MODE_PYMOOSE = 1 # The command line is a python shell
    

	



# 
# mooseglobals.py ends here
