# CaPool.py --- 
# 
# Filename: capool.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Wed Apr 22 22:21:11 2009 (+0530)
# Version: 
# Last-Updated: Sat May 26 10:23:30 2012 (+0530)
#           By: subha
#     Update #: 180
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Implements the Ca2+ pool
# 
# 

# Change log:
# 
# 
# 
# 

# Code:

import moose

class CaPool(moose.CaConc):
    _prototypes = {}
    def __init__(self, path):
        if moose.exists(path):
            moose.CaConc.__init__(self, path)
            return
        moose.CaConc.__init__(self, path)
        self.CaBasal = 0.0
        self.ceiling = 1e6
        self.floor = 0.0
        

def initCaPoolPrototypes(libpath='/library'):
    if CaPool._prototypes:
        return CaPool._prototypes
    path = '%s/CaPool' % (libpath)
    CaPool._prototypes['CaPool'] = CaPool(path)
    print 'Created CaConc prototype:', path
    return CaPool._prototypes


# 
# capool.py ends here
