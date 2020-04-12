# -*- coding: utf-8 -*-

# Bring everything from moose.py to global namespace.
# IMP: It will overwrite any c++ function with the same name.  We can override
# some C++ here.
from moose._moose import *
from moose.moose import *

from moose.server import *

# SBML and NML2 support.
from moose.model_utils import *

# Import moose test.
from moose.moose_test import test

__version__ = _moose.__version__
