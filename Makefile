#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2003- 2006 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/
#######################################################################
# NOTE:
# This Makefile is compatible with _GNU Make_.
# This does not work with nmake or borland make.
# You may have to specify some variables when calling gnu make as 
# described in the comments below. The defaults should work on most
# Unix clones. 
########################################################################

# Linux compilation:
# We recommend two levels of compilation: either full debug, with gdb,
# unit tests and all the rest, or an optimized version for production
# simulations, without any unit tests or assertions. If you want some
# intermediate levels, edit the flags.
######################################################################
#
#     ADDITIONAL COMMANDLINE VARIABLES FOR MAKE
#
######################################################################     
# The variable BUILD determines if it should be optimized (release)
# or a debug version (default).
# make can be run with a command line parameter like below:
# make clean BUILD=debug
# make BUILD=debug
# another option is to define BUILD as an environment variable:
# export BUILD=debug
# make clean
# make
#
# There are some more variables which just need to be defined for 
# controlling the compilation and the value does not matter. These are:
#
# USE_GSL - use GNU Scientific Library for integration in kinetic simulations
# 
# USE_READLINE - use the readline library which provides command history and 
# 		better command line editing capabilities
# 
# GENERATE_WRAPPERS - useful for python interface developers. The binary created 
# 		with this option looks for a directory named 'generated' in the
# 		working directory and creates a wrapper class ( one .h file 
# 		and a .cpp file ) and partial code for the swig interface file
# 		(pymoose.i). These files with some modification can be used for
# 		generating the python interface using swig.
#
# USE_MPI - compile with support for parallel computing through MPICH library
#

# BUILD (= debug, release)
ifndef BUILD
BUILD=debug
endif

#If using mac uncomment the following lines
# PLATFORM=mac
#export PLATFORM

# Get the processor architecture - i686 or x86_64
# All these should be taken care of in a script, not in the 
# Makefile. But we are
ifndef MACHINE
MACHINE=i686 
endif
# We are assuming all non-win32 systems to be POSIX compliant
# and thus have the command uname for getting Unix system name
ifneq ($(OSTYPE),win32)
MACHINE=$(shell uname -m)
PLATFORM := $(shell uname -s)
endif

# Get the python version
ifneq ($(OSTYPE),win32)
PYTHON_VERSION := $(subst ., ,$(lastword $(shell python --version 2>&1)))
PYTHON_VERSION_MAJOR := $(word 1,${PYTHON_VERSION})
PYTHON_VERSION_MINOR := $(word 2,${PYTHON_VERSION})
INSTALLED_PYTHON := python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}
endif

# Debug mode:
ifeq ($(BUILD),debug)
CXXFLAGS = -g -pthread -Wall -Wno-long-long -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER
USE_GSL = 1
endif
# Optimized mode:
ifeq ($(BUILD),release)
CXXFLAGS  = -O3 -Wall -Wno-long-long -pedantic -DNDEBUG -DUSE_GENESIS_PARSER
USE_GSL = 1
endif
# Profiling mode:
ifeq ($(BUILD),profile)
CXXFLAGS  = -O3 -pg -Wall -Wno-long-long -pedantic -DNDEBUG -DUSE_GENESIS_PARSER  
USE_GSL = 1
endif
# Threading mode:
ifeq ($(BUILD),thread)
CXXFLAGS  = -O3 -pthread -Wall -Wno-long-long -pedantic -DNDEBUG -DUSE_GENESIS_PARSER  
USE_GSL = 1
endif

# MPI mode:
ifeq ($(BUILD),mpi)
CXXFLAGS  = -g -pthread -Wall -Wno-long-long -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER
USE_MPI = 1
USE_GSL = 1
endif

# optimized MPI mode:
ifeq ($(BUILD),ompi)
CXXFLAGS  = -O3 -pthread -Wall -Wno-long-long -pedantic -DNDEBUG -DUSE_GENESIS_PARSER
USE_MPI = 1
USE_GSL = 1
endif

# optimised mode but with unit tests.
ifeq ($(BUILD),odebug)
CXXFLAGS = -O3 -pthread -Wall -Wno-long-long -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER
USE_GSL = 1
endif

# including SMOLDYN
ifdef USE_SMOLDYN
CXXFLAGS = -g -pthread -Wall -Wno-long-long -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER
USE_GSL = 1
endif

##########################################################################
#
# MAC OS X compilation, Debug mode:
ifeq ($(PLATFORM),Darwin)
CXXFLAGS += -DMACOSX # GCC compiler also sets __APPLE__ for Mac OS X which can be used instead
CXXFLAGS += -Wno-deprecated -force_cpusubtype_ALL -mmacosx-version-min=10.4
endif
# Use the options below for compiling on GCC4.1
# GNU C++ 4.1 and newer might need -ffriend-injection
#
#CXXFLAGS  =	-g -Wall -pedantic -DDO_UNIT_TESTS -ffriend-injection -DUSE_GENESIS_PARSER

# Insert the svn revision no. into the code as a preprocessor macro.
# Only for release versions we want to pass SVN=0 to make.
ifndef SVN
SVN?=1
endif
ifneq ($(SVN),0)
SVN_REVISION=$(shell svnversion)
ifneq ($(SVN_REVISION),export)
CXXFLAGS+=-DSVN_REVISION=\"$(SVN_REVISION)\"
endif
endif


# Libraries are defined below.
SUBLIBS = 
LIBS =	-lm -lpthread -L/usr/lib -L/usr/local/lib
#LIBS = 	-lm
#ifeq ($(BUILD),thread)
#LIBS += -lpthread
#endif

# For 64 bit Linux systems add paths to 64 bit libraries 
ifeq ($(PLATFORM),Linux)
CXXFLAGS += -DLINUX
ifeq ($(MACHINE),x86_64)
LIBS+= -L/lib64 -L/usr/lib64
endif
endif

##########################################################################
#
# Developer options (Don't try these unless you are writing new code!)
##########################################################################
# For parallel (MPI) version:
ifdef USE_MUSIC
USE_MPI = 1		# Automatically enable MPI if USE_MUSIC is on
CXXFLAGS += -DUSE_MUSIC
LIBS += -lmusic
endif

# The -DMPICH_IGNORE_CXX_SEEK flag is because of a bug in the
# MPI-2 standard. Enabled by default because it use crops up
# often enough. You won't need if if you are not using MPICH, or
# if your version of MPICH has fixed the issue.
ifdef USE_MPI
# CXXFLAGS += -DUSE_MPI
CXXFLAGS += -DUSE_MPI -DMPICH_IGNORE_CXX_SEEK
endif

#use this for readline library
#CXXFLAGS = -g -Wall -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER -DUSE_READLINE


# To use GSL, pass USE_GSL=true ( anything on the right will do) in make command line
ifdef USE_GSL
LIBS+= -L/usr/lib -lgsl -lgslcblas
CXXFLAGS+= -DUSE_GSL
endif

# To use Smoldyn, pass USE_SMOLDYN=true ( anything on the right will do) in make command line
ifdef USE_SMOLDYN
#LIBS+= -L/usr/local/lib -lsmoldyn
CXXFLAGS+= -DUSE_SMOLDYN
SMOLDYN_DIR = smol
SMOLDYN_LIB = smol/_smol.o /usr/local/lib/libsmoldyn.a
LIBS += -lsmoldyn
endif

# To compile with readline support pass USE_READLINE=true in make command line
ifdef USE_READLINE
LIBS+= -lreadline
CXXFLAGS+= -DUSE_READLINE
endif

# To compile with curses support (terminal aware printing) pass USE_CURSES=true in make command line
ifdef USE_CURSES
LIBS += -lcurses
CXXFLAGS+= -DUSE_CURSES
endif

ifdef USE_MUSIC
	MUSIC_DIR = music
	MUSIC_LIB = music/music.o
endif

# Here we automagically change compilers to deal with MPI.
ifdef USE_MPI
	CXX = mpicxx
#	CXX = /usr/local/mvapich2/bin/mpicxx
#	PARALLEL_DIR = parallel
#	PARALLEL_LIB = parallel/parallel.o
else
	CXX = g++
#	CXX = CC	# Choose between Solaris CC and g++ on a Solaris machine
endif

LD = ld

SUBDIR = \
	basecode \
	msg \
	shell \
	biophysics\
	randnum\
	scheduling\
	builtins\
	kinetics \
	ksolve \
	regressionTests \
	utility \
	geom \
	mesh \
	manager \
	$(SMOLDYN_DIR) \


# Used for 'make clean'
CLEANSUBDIR = $(SUBDIR) $(PARALLEL_DIR) pymoose

OBJLIBS =	\
	basecode/_basecode.o \
	msg/_msg.o \
	shell/_shell.o \
	biophysics/_biophysics.o \
	randnum/_randnum.o \
	scheduling/_scheduling.o \
	builtins/_builtins.o \
	kinetics/_kinetics.o \
	ksolve/_ksolve.o \
	regressionTests/_rt.o \
	utility/_utility.o \
	geom/_geom.o \
	mesh/_mesh.o \
	manager/_manager.o \
	$(SMOLDYN_LIB) \

export CXX
export CXXFLAGS
export LD
export LIBS
export USE_GSL

moose: libs $(OBJLIBS) $(PARALLEL_LIB)
	$(CXX) $(CXXFLAGS) $(OBJLIBS) $(PARALLEL_LIB) $(LIBS) -o moose
	@echo "Moose compilation finished"

libmoose.so: libs
	$(CXX) -G $(LIBS) -o libmoose.so
	@echo "Created dynamic library"

# There are some unix/gcc specific paths here. Should be cleaned up later.
pymoose: CXXFLAGS += -DPYMOOSE -fPIC -fno-strict-aliasing -I/usr/include/${INSTALLED_PYTHON} # Should be updated according to location of user include directory
pymoose: SUBDIR += pymoose
pymoose: OBJLIBS += pymoose/pymoose.o
pymoose: LIBS += -l${INSTALLED_PYTHON}
pymoose: python/moose/_moose.so
python/moose/_moose.so: libs $(OBJLIBS)
	$(CXX) -shared $(LDFLAGS) $(CXXFLAGS) -o $@ $(OBJLIBS) $(LIBS)

libs:
	@(for i in $(SUBDIR) $(PARALLEL_DIR); do $(MAKE) -C $$i; done)
	@echo "All Libs compiled"

mpp: preprocessor/*.cpp preprocessor/*.h
	@( rm -f mpp; cd preprocessor; make CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS)"; ln mpp ..; cd ..)

default: moose mpp

clean:
	@(for i in $(CLEANSUBDIR) ; do $(MAKE) -C $$i clean;  done)
	-rm -rf moose mpp core.* DOCS/html python/moose/*.so python/moose/*.pyc  
