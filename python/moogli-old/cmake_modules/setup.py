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


"""setup-moogli.py: 

    Script to install moogli extension.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import os
from distutils.core import setup
 
cwd_ = os.getcwd()

setup( name             =   'moogli'
     , version          =   '1.0'
     , author           =   'Aviral Goel'
     , author_email     =   'aviralg@ncbs.res.in'
     , maintainer       =   'Aviral Goel'
     , maintainer_email =   'aviralg@ncbs.res.in'
     , url              =   ''
     , download_url     =   ''
     , description      =   ''
     , long_description =   ''
     , classifiers      =   [ 'Development Status :: Alpha'
                            , 'Environment :: GUI'
                            , 'Environment :: Desktop'
                            , 'Intended Audience :: End Users/Desktop'
                            , 'Intended Audience :: Computational Neuroscientists'
                            , 'License :: GPLv3'
                            , 'Operating System :: Linux :: Ubuntu'
                            , 'Programming Language :: Python'
                            , 'Programming Language :: C++'
                            ]
     , license          =   'GPLv3'
     , packages         =   [ 'moogli' ]
     , package_dir      =   { 'moogli' : '.' }
     , package_data     =   { 'moogli' : [ '_moogli.so' ] }
     )
