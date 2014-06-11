# This file is part of MOOSE simulator: http://moose.ncbs.res.in.

# MOOSE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# MOOSE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with MOOSE.  If not, see <http://www.gnu.org/licenses/>.


"""test_mumbl.py: 

    A test script to test MUMBL support in MOOSE.

Last modified: Mon Jun 09, 2014  03:42PM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
import moose
import moose.neuroml as nml
import moose.mumbl as mumbl
import moose.utils as utils

def main():
    utils.parser
    print dir(nml)
    nml.loadNeuroML_L123('./two_cells_nml_1.8/two_cells.nml')
    mumbl.loadMumbl("./two_cells_nml_1.8/mumbl.xml")
    table = utils.recordTarget('/tableA', '/neuroml/cells/cellA/Dend_37_41', 'vm')
    moose.reinit()
    moose.start(0.1)
    utils.plotTable(table)
    
if __name__ == '__main__':
    main()
