# MOOSEClasses.py
# 
# Filename: MOOSEClasses.py
# Description: This file contains the MOOSE class names divided 
#       into categories for easy incorporation into the toolbox.
# Author: subhasis ray
# Maintainer: 
# Created: Sun Apr 12 14:05:01 2009 (+0530)
# Version: 
# Last-Updated: Sun Apr 12 14:11:35 2009 (+0530)
#           By: subhasis ray
#     Update #: 8
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

class MOOSEClasses:
    """This class roughly reflects the directory structure in MOOSE source code.

    The primary purpose of this class is to divide the MOOSE classes
    into categories.  Each category will form a page in the GUI
    toolbox and the list of classes under that category will be
    displayed on that page. The list "category" is used to order the
    pages in toolbox."""
    categories = ["base", "builtins", "biophysics", "kinetics", "device"]
    class_dict = {
        "base":[
                "Neutral", 
                "ClockTick"],
        "biophysics": [
                "BinSynchan",
                "BioScan",
                "CaConc",
                "Cell",
                "Compartment",
                "DifShell",
                "GHK",
                "HHChannel2D",
                "HHChannel",
                "HHGate2D",
                "HHGate",
                "IntFire",
                "IzhikevichNrn",
                "Mg_block",
                "MMPump",
                "Nernst",
                "RandomSpike",
                "script_out",
                "SpikeGen",
                "StochSynchan",
                "SymCompartment",
                "SynChan",
                "TauPump",
                "TestBiophysics"],
        "builtins":[
                "AscFile",
                "Calculator",
                "Interpol2D",
                "Interpol",
                "Table",
                "TimeTable"],
        "device": [
                "DiffAmp",
                "PIDController",
                "PulseGen",
                "RC"],
        "kinetics": [
                "CylPanel",
                "DiskPanel",
                "Enzyme",
                "Geometry",
                "HemispherePanel",
                "KinCompt",
                "KinPlaceHolder",
                "MathFunc",
                "Molecule",
                "Panel",
                "Particle",
                "Reaction",
                "RectPanel",
                "SmoldynHub",
                "SpherePanel",
                "Surface",
                "TriPanel"]}



# 
# MOOSEClasses.py ends here
