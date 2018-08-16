# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

# Bring moose.py functions into global namespace.
# Try to set the backend. This must be called before any instance of matplotlib
# is imported i.e. import 'moose' after this.
import matplotlib
try:
    matplotlib.use( 'TkAgg' )
except Exception as e:
    try:
        matplotlib.use( 'Qt5Agg')
    except Exception as e:
        try:
            matplotlib.use( 'Qt4Agg')
        except Exception as e:
            pass

from moose.moose import *

__version__ = version( )
