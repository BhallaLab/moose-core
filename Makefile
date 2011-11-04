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
# make can be run with a command line parameter like below:
# 		make clean
# 		make BUILD=debug USE_SBML=0 USE_MPI=1
# 
# another option is to define BUILD as an environment variable:
# 		export BUILD=debug
# 		export USE_SBML=0
# 		export USE_MPI=1
# 		make clean
# 		make
#
# There are a few variables whose value you can set to control compilation.
# Choose the libraries you want by setting the USE_* flags to 0 or 1 (to exclude
# or include the library). The variables you can set are:
# 
# BUILD (default value: release) - If this variable is set to 'release' (default),
# 		moose will be compiled in optimized mode. If it is set to 'debug', then 
# 		debug symbols will be included and compiler optimizations will not be used.
#
# USE_GSL (default value: 1) - use GNU Scientific Library for integration in
# 		kinetic simulations.
#		
# USE_SBML (default value: 1) - compile with support for the Systems Biology
# 		Markup Language (SBML). This allows you to read and write chemical 
# 		kinetic models in the simulator-indpendent SBML format.
# 
# USE_NEUROML (default value: 0) - compile with support for the NeuroML. This 
#		allows you to read neuronal models in the NeuroML format.
#		Look in external/neuroML_src/README for the extra steps needed 
#		to add the libraries & headers.
#
# USE_READLINE (default value: 1) - use the readline library which provides
# 		command history and better command line editing capabilities
#
# USE_MPI (default value: 0) - compile with support for parallel computing through
# 		MPICH library
# 
# USE_MUSIC (default value: 0) - compile with MUSIC support. The MUSIC library 
# 		allows runtime exchange of information between simulators.
#
# USE_CURSES (default value: 0) - To compile with curses support (terminal aware
# 		printing)
# 
# USE_GL (default value: 0) - To compile with OpenSceneGraph support to enable the MOOSE
# 		elements 'GLcell', 'GLview'.
#
# GENERATE_WRAPPERS (default value: 0) - useful for python interface developers.
# 		The binary created with this option looks for a directory named
# 		'generated' in the working directory and creates a wrapper class
# 		(one .h file and a .cpp file ) and partial code for the swig interface
# 		file (pymoose.i). These files with some modification can be used for
# 		generating the python interface using swig.
#

# Default values for flags. The operator ?= assigns the given value only if the
# variable is not already defined.

# BUILD (= debug, release)
BUILD?=release
USE_GSL?=1
USE_SBML?=1
USE_NEUROML?=0
USE_READLINE?=1
USE_MPI?=0
USE_MUSIC?=0
USE_CURSES?=0
USE_GL?=0
GENERATE_WRAPPERS?=0
SVN?=0
export BUILD
export USE_GSL
export USE_SBML
export USE_NEUROML
export USE_READLINE
export USE_MPI
export USE_MUSIC
export USE_CURSES
export USE_GL
export GENERATE_WRAPPERS

# PLATFORM (= Linux, win32, mac)
#If using mac uncomment the following lines
PLATFORM=Linux
#export PLATFORM

# Get the processor architecture - i686 or x86_64
# All these should be taken care of in a script, not in the 
# Makefile. But we are
MACHINE?=i686 

# We are assuming all non-win32 systems to be POSIX compliant
# and thus have the command uname for getting Unix system name
ifneq ($(OSTYPE),win32)
MACHINE=$(shell uname -m)
endif

# Debug mode:
ifeq ($(BUILD),debug)
CXXFLAGS = -g -Wall -Wno-long-long -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER
endif
# Optimized mode:
ifeq ($(BUILD),release)
CXXFLAGS  = -O3 -Wall -Wno-long-long -pedantic -DNDEBUG -DUSE_GENESIS_PARSER 
endif
# Insert the svn revision no. into the code as a preprocessor macro
ifneq ($(SVN),0)
SVN_REVISION=$(shell svnversion)
ifneq ($(SVN_REVISION),export)
CXXFLAGS+=-DSVN_REVISION=\"$(SVN_REVISION)\"
endif
endif

##########################################################################
#
# MAC OS X compilation, Debug mode:
ifeq ($(PLATFORM),mac)
CXXFLAGS += -Wno-deprecated -force_cpusubtype_ALL -mmacosx-version-min=10.4
endif
# Use the options below for compiling on GCC4.1
# GNU C++ 4.1 and newer might need -ffriend-injection
#
#CXXFLAGS  =	-g -Wall -pedantic -DDO_UNIT_TESTS -ffriend-injection -DUSE_GENESIS_PARSER

##########################################################################
#
# Don't mess with stuff below!
#
##########################################################################


# Libraries are defined below. For now we do not use threads.
SUBLIBS = 
#LIBS =	-lm -lpthread
LIBS = 	-lm 
##########################################################################
#
# Developer options (Don't try these unless you are writing new code!)
#
# For generating python interface:
# Do remember that you have to create a directory named "generated" 
# in the working directory of moose. Also you have to do some editing 
# to get the generated code to work. 
# Although this binary of MOOSE is verbose in its complaints, is completely harmless 
# except for the overhead of  checks for the existence of a few files at startup.
ifeq ($(GENERATE_WRAPPERS),1)
CXXFLAGS += -DGENERATE_WRAPPERS
endif

# For parallel (MPI) version:
ifeq ($(USE_MUSIC),1)
USE_MPI = 1 # Automatically enable MPI if USE_MUSIC is on (doesn't seem to work though.)
CXXFLAGS += -DUSE_MUSIC
LIBS += -lmusic
MUSIC_DIR = music
MUSIC_LIB = music/music.o
endif

# The -DMPICH_IGNORE_CXX_SEEK flag is because of a bug in the
# MPI-2 standard. Enabled by default because it use crops up
# often enough. You won't need if if you are not using MPICH, or
# if your version of MPICH has fixed the issue.
ifeq ($(USE_MPI),1)
# CXXFLAGS += -DUSE_MPI
CXXFLAGS += -DUSE_MPI -DMPICH_IGNORE_CXX_SEEK
CXX = mpicxx
PARALLEL_DIR = parallel
PARALLEL_LIB = parallel/parallel.o
endif

#use this for readline library
#CXXFLAGS = -g -Wall -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER -DUSE_READLINE


# To use GSL, pass USE_GSL=1 in make command line
ifeq ($(USE_GSL),1)
LIBS+= -lgsl -lgslcblas
CXXFLAGS+= -DUSE_GSL 
endif

# To use SBML, pass USE_SBML=1 in make command line
ifeq ($(USE_SBML),1)
LIBS+= -lsbml
CXXFLAGS+=-DUSE_SBML 
LDFLAGS += -L/usr/lib
SBML_DIR = sbml_IO
SBML_LIB = sbml_IO/sbml_IO.o 
endif

# To use NeuroML, pass USE_NeuroML=1 in make command line
ifeq ($(USE_NEUROML),1)
LIBS+= -lxml2 -lneuroml
LDFLAGS+= -Lexternal/neuroML_src
CXXFLAGS+=-DUSE_NEUROML
NEUROML_DIR = neuroML_IO
NEUROML_LIB = neuroML_IO/neuroML_IO.o
LIBNEUROML_SRC = external/neuroML_src
LIBNEUROML_DYNAMIC = external/neuroML_src/libneuroml.so
LIBNEUROML_STATIC = external/neuroML_src/libneuroml.a
endif

# To compile with readline support pass USE_READLINE=1 in make command line
ifeq ($(USE_READLINE),1)
LIBS+= -lreadline -lncurses
CXXFLAGS+= -DUSE_READLINE 
endif

# To compile with curses support (terminal aware printing) pass USE_CURSES=1 in make command line
ifeq ($(USE_CURSES),1)
LIBS += -lcurses
CXXFLAGS+= -DUSE_CURSES
endif

# To compile with OpenSceneGraph support and enable 'GLcell', 'GLview' pass USE_GL=1 in make command line
ifeq ($(USE_GL),1)
	LIBS += -losg -losgDB -lOpenThreads -lboost_serialization
	LDFLAGS += -L/usr/local/lib 
	CXXFLAGS += -DUSE_GL -I. -Ibasecode
	GL_DIR = gl/src
	GLCELL_LIB = gl/src/GLcell.o
	GLVIEW_LIB = gl/src/GLview.o gl/src/GLshape.o
endif

# For mac with USE_GL, force 32-bit architecture because OSG doesn't fully build in 64-bit yet
ifeq ($(PLATFORM),mac)
ifeq ($(USE_GL),1)
CXXFLAGS += -arch i386
endif
endif

# For 64 bit Linux systems add paths to 64 bit libraries 
ifeq ($(OSTYPE),Linux)
ifeq ($(MACHINE),x86_64)
LDFLAGS +=-L/lib64 -L/usr/lib64
endif
endif


LD = ld

SUBDIR = basecode connections maindir genesis_parser shell element scheduling \
	biophysics hsolve kinetics ksolve builtins utility \
	randnum robots device $(GL_DIR) $(SBML_DIR) $(NEUROML_DIR) $(PARALLEL_DIR) $(MUSIC_DIR) 

# Used for 'make clean'
CLEANSUBDIR = $(SUBDIR) gl/src sbml_IO neuroML_IO parallel music pymoose $(LIBNEUROML_SRC)

OBJLIBS =	\
	basecode/basecode.o \
	connections/connections.o \
	maindir/maindir.o \
	genesis_parser/SLI.o \
	element/element.o \
	shell/shell.o \
	utility/utility.o \
	randnum/randnum.o	\
	scheduling/scheduling.o \
	biophysics/biophysics.o \
	hsolve/hsolve.o \
	kinetics/kinetics.o \
	ksolve/ksolve.o \
	builtins/builtins.o \
	robots/robots.o \
	device/device.o \
	$(GLCELL_LIB) \
	$(GLVIEW_LIB) \
	$(SBML_LIB) \
	$(NEUROML_LIB) \
	$(PARALLEL_LIB) \
	$(MUSIC_LIB)

export CXX
export CXXFLAGS
export LD
export LDFLAGS
export LIBS

moose: libs $(OBJLIBS) $(LIBNEUROML_STATIC)
	$(CXX) $(LDFLAGS) $(CXXFLAGS) $(OBJLIBS) $(LIBS) -o moose 
	@echo "Moose compilation finished"

libmoose.so: libs
	$(CXX) -G $(LIBS) -o libmoose.so
	@echo "Created dynamic library"

.PHONEY : pymoose

pymoose: CXXFLAGS += -DPYMOOSE -fPIC -I/usr/include/python2.6
pymoose: SUBDIR += pymoose	
pymoose: OBJLIBS := pymoose/pymoose.o $(OBJLIBS)
pymoose: LIBS += -lpython2.6
pymoose: python/moose/_moose.so	

python/moose/_moose.so: libs $(OBJLIBS) $(LIBNEUROML_DYNAMIC) 
	$(CXX) -shared $(LDFLAGS) $(CXXFLAGS) -o $@ $(OBJLIBS) $(LIBS)
	cp pymoose/moose.py ./python/moose/

$(LIBNEUROML_DYNAMIC): 
	$(MAKE) -C $(LIBNEUROML_SRC) TYPE=dynamic

$(LIBNEUROML_STATIC):
	$(MAKE) -C $(LIBNEUROML_SRC) TYPE=static


libs:
	@echo "Compiling with flags:"
	@echo "	BUILD:" $(BUILD)
	@echo "	USE_GSL:" $(USE_GSL)
	@echo "	USE_SBML:" $(USE_SBML)
	@echo "	USE_NEUROML:" $(USE_NEUROML)
	@echo "	USE_READLINE:" $(USE_READLINE)
	@echo "	USE_MPI:" $(USE_MPI)
	@echo "	USE_MUSIC:" $(USE_MUSIC)
	@echo "	USE_CURSES:" $(USE_CURSES)
	@echo "	USE_GL:" $(USE_GL)
	@echo "	SVN_REVISION:" $(SVN_REVISION)
	@echo "	GENERATE_WRAPPERS:" $(GENERATE_WRAPPERS)
	@echo "	LDFLAGS:" $(LDFLAGS)
	@(for i in $(SUBDIR); do $(MAKE) -C $$i; done)
	@echo "All Libs compiled"


default: moose 

clean:
	@(for i in $(CLEANSUBDIR) ; do $(MAKE) -C $$i clean;  done)
	-rm -rf moose mpp core.* DOCS/html *.so moose.py *.pyc

