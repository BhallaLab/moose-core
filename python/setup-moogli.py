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


"""setup.py: 

    Script to install python targets.

Last modified: Sat Jan 18, 2014  05:01PM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, Aviral Goel"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Aviral Geol, Dilawar Singh"
__email__            = "aviralg@ncbs.res.in"
__status__           = "Development"

import os
from setuptools import setup

setup(
        name='moogli',
        version='1.0',
        description='Visualizer for nerual simulation',
        author='Aviral Goel',
        author_email='aviralg@ncbs.res.in',
        url='http://moose.ncbs.res.in/moogli',
        options={'build' : {'build_base' : '/tmp' } },
        packages=[ 'moogli'] ,
        package_dir = { 'moogli' : 'moogli' },
        package_data = { 'moogli' : ['_moogli.so'] },
        #install_requires = [ 'sip' ],
    ) 
