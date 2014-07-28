
"""setup.py: This scripts prepare MOOSE for PyPI.

Last modified: Mon Jul 28, 2014  12:52AM

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
import sys

from distutils.core import setup, Command
from distutils.command.install import install as _install

import distutils.spawn as ds

class ConfigureCommand(Command):
    """Configure MOOSE """
    user_options = []

    def initialize_options(self):
        self.cwd = os.getcwd()
        self.new_dir = os.path.join(os.path.split(__file__)[0], 'buildMooseUsingCmake')
    
    def finalize_options(self):
        pass

    def run(self):
        if ds.find_executable('cmake') is None:
            msg = [ "Error: Unable to configure MOOSE."
                    , " "
                    , " This application relies on CMake build tool (www.cmake.org)"
                    , " Once cmake is installed, you can continue "
                    ]
            print("\n".join(msg))
            sys.exit(0)

        print("Configuring MOOSE using CMake")
        os.chdir(self.new_dir)
        try:
            ds.spawn(['cmake', '..'])
        except ds.DistutilsExecError:
            print("Error: error occurred while running CMake to configure MOOSE.")
            os.chdir(self.cwd)
            sys.exit(-1)
        os.chdir(self.cwd)

class BuildCommand(Command):
    user_options = []
    def initialize_options(self):
        self.cwd = os.getcwd()
        self.new_dir = os.path.join(os.path.split(__file__)[0], 'buildMooseUsingCmake')
    
    def finalize_options(self):
        pass

    def run(self):
        print("Building PyMOOSE")
        os.chdir(self.new_dir)
        try:
            ds.spawn(['make'])
        except ds.DistutilsExecError as e:
            print("Can't build MOOSE")
            print(e)
            os.chdir(self.cwd)
            sys.exit(-1)
        os.chdir(self.cwd)

##
# @brief FUnction to read a file.
#
# @param fname Name of the file.
#
# @return  A string content of the file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

name          = 'moose'
version       = '3.0'
description   = (
        'MOOSE is the Multiscale Object-Oriented Simulation Environment. '
        'It is the base and numerical core for large, detailed simulations '
        'including Computational Neuroscience and Systems Biology.' )
url           = 'http://moose.ncbs.res.in/'


setup(
        name = name
        , version = version 
        , author = [ "Upinder Bhalla et. al." ]
        , author_email = "bhalla@ncbs.res.in"
        , description = description 
        , license = "LGPL"
        , keywords = "neural simulation"
        , url = url
        , packages = []
        , long_description = read('README')
        , cmdclass = { 'configure' : ConfigureCommand 
            , 'build' : BuildCommand 
            }
        )
