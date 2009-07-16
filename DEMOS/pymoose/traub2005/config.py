# config.py --- 
# 
# Filename: config.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Fri Apr 17 14:36:30 2009 (+0530)
# Version: 
# Last-Updated: Wed May  6 17:29:53 2009 (+0530)
#           By: subhasis ray
#     Update #: 12
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# This contains global variables used in other modules.
# 
# 

# Change log:
# 
# 
# 
# 
# Code:

import sys
sys.path.append('/home/src/sim/cortical/py')
import moose
context = moose.PyMooseBase.getContext()
lib = moose.Neutral('/library')
root = moose.Neutral("/")

simdt = 1e-6
plotdt = 1e-6
vmin = -120e-3
vmax = 40e-3
ndivs = 640
dv = (vmax - vmin) / ndivs


# 
# config.py ends here
