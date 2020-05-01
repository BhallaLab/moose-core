# savemodel.py ---

# Filename: savemodel.py
# Description:
# Author: Subhasis Ray
# Maintainer:
# Created: Wed Oct 29 00:08:50 2014 (+0530)
# Version:
# Last-Updated:Thr Dec 23 16:31:00 2015 (+0530)
#           By:
#     Update #: 0
# URL:
# Keywords:
# Compatibility:

import os
import traceback
import moose

cwd = os.path.dirname( os.path.realpath( __file__ ) )

def test_kkit():
    """ 
    The script demonstates to convert Chemical (Genesis) file back to Genesis 
    file using moose 
    """
    moose.loadModel(os.path.join( cwd, '../data/reaction.g'), '/model')
    written = moose.writeKkit('/model', 'testsave.g')
    print( written )

if __name__ == '__main__':
    test_kkit()
