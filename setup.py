# -*- coding: utf-8 -*-
# This script can also be called directly to build and install the pymoose
# module.
#
# Alternatively, you can use cmake build system which provides finer control
# over the build. This script is called by cmake to install the python module. 
# 
# This script is compatible with python2.7 and python3+. Therefore use of
# super() is commented out.
#
# NOTES:
#  * Python2
#   - Update setuptools using `python2 -m pip install setuptools --upgrade --user'.

__author__     = "Dilawar Singh"
__copyright__  = "Copyright 2019-, Dilawar Singh"
__maintainer__ = "Dilawar Singh"
__email__      = "dilawar.s.rajput@gmail.com"

import os
import sys
import subprocess
import datetime

try:
    cmakeVersion = subprocess.check_output(["cmake", "--version"])
except Exception as e:
    print("[ERROR] cmake is not found. Please install cmake.")
    quit(-1)

# See https://docs.python.org/3/library/distutils.html
# setuptools is preferred over distutils. And we are supporting python3 only.
from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext  as _build_ext
import subprocess

# Global variables.
sdir_ = os.path.dirname(os.path.realpath(__file__))
stamp = datetime.datetime.now().strftime('%Y%m%d')
builddir_ = os.path.join(sdir_, '%s_build_%s' % (sys.version_info[0], stamp))

if not os.path.exists(builddir_):
    os.makedirs(builddir_)

numCores_ = 2
try:
    # Python3 only.
    numCores_ = os.cpu_count()
except Exception:
    pass

version_ = '3.2.dev%s' % stamp

# importlib is available only for python3. Since we build wheels, prefer .so
# extension. This way a wheel built by any python3.x will work with any python3.

class CMakeExtension(Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        Extension.__init__(self, name, sources=[])

class TestCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print("[INFO ] Running tests... ")
        os.chdir(builddir_)
        self.spawn(["ctest", "--output-on-failure", '-j2'])
        os.chdir(sdir_)

class build_ext(_build_ext):
    user_options = [('with-boost', None, 'Use Boost Libraries (OFF)')
            , ('with-gsl', None, 'Use Gnu Scienfific Library (ON)')
            , ('debug', None, 'Build moose in debugging mode (OFF)')
            , ('no-build', None, 'DO NOT BUILD. (for debugging/development)')
            ] + _build_ext.user_options

    def initialize_options(self):
        # Initialize options.
        self.with_boost = 0
        self.with_gsl = 1
        self.debug = 0
        self.no_build = 0
        self.cmake_options = {}
        #  super().initialize_options()
        _build_ext.initialize_options(self)

    def finalize_options(self):
        # Finalize options.
        #  super().finalize_options()
        _build_ext.finalize_options(self)
        self.cmake_options['PYTHON_EXECUTABLE'] = os.path.realpath(sys.executable)
        if self.with_boost:
            self.cmake_options['WITH_BOOST'] = 'ON'
            self.cmake_options['WITH_GSL'] = 'OFF'
        if self.debug:
            self.cmake_options['CMAKE_BUILD_TYPE'] = 'Debug'
        else:
            self.cmake_options['CMAKE_BUILD_TYPE'] = 'Release'

    def run(self):
        if self.no_build:
            return
        for ext in self.extensions:
            self.build_cmake(ext)
        #  super().run()
        _build_ext.run(self)

    def build_cmake(self, ext):
        global numCores_
        global sdir_
        print("\n==========================================================\n")
        print("[INFO ] Building pymoose in %s ..." % builddir_)
        cmake_args = []
        for k, v in self.cmake_options.items():
            cmake_args.append('-D%s=%s' % (k,v))
        os.chdir(str(builddir_))
        self.spawn(['cmake', str(sdir_)] + cmake_args)
        if not self.dry_run: 
            self.spawn(['make', '-j%d'%numCores_]) 
        os.chdir(str(sdir_))

with open(os.path.join(sdir_,  "README.md")) as f:
    readme = f.read()

setup(
    name="pymoose",
    version=version_,
    description= 'Python scripting interface of MOOSE Simulator (https://moose.ncbs.res.in)',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='MOOSERes',
    author_email='bhalla@ncbs.res.in',
    maintainer='Dilawar Singh',
    maintainer_email='dilawars@ncbs.res.in',
    url='http://moose.ncbs.res.in',
    packages=[
        'rdesigneur', 'moose', 'moose.SBML', 'moose.genesis', 'moose.neuroml',
        'moose.neuroml2', 'moose.chemUtil', 'moose.chemMerge'
    ],
    # python2 specific version here as well.
    install_requires=['numpy'],
    package_dir={
        'moose': os.path.join(sdir_, 'python', 'moose'),
        'rdesigneur': os.path.join(sdir_, 'python', 'rdesigneur')
    },
    package_data={
        'moose': [
            '_moose.so'
            , os.path.join('neuroml2','schema','NeuroMLCoreDimensions.xml')
            , os.path.join('chemUtil', 'rainbow2.pkl')
        ]
    },
    ext_modules=[CMakeExtension('pymoose')],
    cmdclass={'build_ext': build_ext, 'test': TestCommand},
)
