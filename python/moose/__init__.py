import moose._moose

# Bring everything from c++ module to global namespace. 
from moose._moose import *

# Bring everything from moose.py to global namespace. 
# IMP: It will overwrite any c++ function with the same name.  We can override
# some C++ here.
if moose._moose.__generated_by__ == "pybind11":
    from moose.moose import *
else:
    from moose.moose_legacy import *

from moose.server import *

# SBML and NML2 support.
from moose.model_utils import *

# create a shorthand for version() call here.
__version__ = version()

# Import moose test.
from moose.moose_test import test
