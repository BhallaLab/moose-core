#!/usr/bin/env python

import distutils.core as dcore

#------------------------------------------------------------------------------
# Metadata
#------------------------------------------------------------------------------
name          = 'moose'
version       = '1.4'
description   = (
	'MOOSE is the Multiscale Object-Oriented Simulation Environment. '
	'It is the base and numerical core for large, detailed simulations '
	'including Computational Neuroscience and Systems Biology.' )
author        = 'Upinder S. Bhalla'
author_email  = 'bhalla@ncbs.res.in'
url           = 'http://moose.ncbs.res.in/'

#------------------------------------------------------------------------------
# Script parameters
#------------------------------------------------------------------------------
#============
# Defaults
#============
mode = 'debug'


#============
# Overriding defaults: command line
#============


#============
# Auto-detect
#============

#------------------------------------------------------------------------------
# Sources
#------------------------------------------------------------------------------
from setup_ import sources

#------------------------------------------------------------------------------
# Include directories
#------------------------------------------------------------------------------
include_dirs = [ 'basecode', 'msg', 'kinetics', 'mesh', 'shell' ]

#------------------------------------------------------------------------------
# Libs, lib dirs
#------------------------------------------------------------------------------
libraries = [ 'gsl', 'gslcblas' ]
library_dirs = []

#------------------------------------------------------------------------------
# Macros
#------------------------------------------------------------------------------
define_macros = [
	( 'PYMOOSE', None ),
	( 'USE_GSL', 1 ),
]

#------------------------------------------------------------------------------
# Compile/Link args
#------------------------------------------------------------------------------
extra_compile_args = [ '-O0', '-Wall', '-pedantic', '-Wno-long-long' ]
extra_link_args = []

#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------
#============
# Parse
#============

#============
# Load
#============

#============
# Save
#============

#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------
moose = dcore.Extension(
	'moose',
	sources,
	define_macros      = define_macros,
	include_dirs       = include_dirs,
	library_dirs       = library_dirs,
	libraries          = libraries,
	extra_compile_args = extra_compile_args,
	extra_link_args    = extra_link_args,
)

dcore.setup(
	name          = name,
	version       = version,
	description   = description,
	author        = author,
	author_email  = author_email,
	url           = url,
	ext_modules   = [ moose ]
)
