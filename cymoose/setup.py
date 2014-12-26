#!/usr/bin/env python

"""setup.py: This is cython interface between moose and python.

Last modified: Sat Jan 18, 2014  05:01PM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@iitb.ac.in"
__status__           = "Development"

import sys
import os
import shutil

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

os.environ['CC'] = "g++"
os.environ['CXX'] = "g++"


# Clean up
moduleName = "cymoose"
#for root, dirs, files in os.walk(".", topdown=False):
#    for name in files:
#        if (name.startswith(moduleName) and not(
#                name.endswith(".pyx") or name.endswith(".pxd") or name.endswith(".cxx") or name.endswith(".hpp")
#                )
#            ):
#            os.remove(os.path.join(root, name))
#for name in dirs:
#    if(name == "build"):
#        shutil.rmtree(name)
#

extensions = [ 
        Extension(moduleName, ["Shell.pyx"]
        , language = "C++"
        , include_dirs = [ 
            "../basecode" 
            , "../msg"
            , "."
            ]
        , extra_compile_args = [
            "-g"
            , "-DCYMOOSE"
            , "-DCYTHON"
            , "-DLINUX"
            ]
        , extra_link_args = ["-L.", "-g"]
        , libraries = [
            "mpi"
            , "moose"
            , "stdc++"
            ]
        )
]

setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules =  cythonize(extensions)
        )
