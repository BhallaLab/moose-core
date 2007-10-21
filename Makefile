#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2003- 2006 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/
# $Id: $
#
# $Log: $
#

# Use the options below for compiling on GCC3. Pick your favourite
# optimization settings.
# Higher optimization levels should use -DNDEBUG to eliminate the
# assertions sprinkled throughout the code
#
#CFLAGS  =	-g -Wall -pedantic -DDO_UNIT_TESTS
#CFLAGS  =	-O3 -Wall -pedantic -DNDEBUG
#CFLAGS  =	-O3 -pg -Wall -pedantic -DNDEBUG

# Use the options below for compiling on GCC4.0

#  On Mac OS X, -Wno-deprecated helps silence some warnings (from GCC 4.0.1)
#CFLAGS  =	-g -Wall -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER -Wno-deprecated

#  For Debian/Ubuntu 6.06, we need to add a few more compiler flags to
#  help it through the genesis parser, which is littered with ifdefs.
#CFLAGS  =	-g -Wall -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER

# !! Avoid the following unless you are a moose python-interface developer. !!
# If you want the python interface wrapper classes to be generated 
# use the following CFLAGS specification.
# Do remember that you have to create a directory named "generated" 
# in the working directory of moose. Also you have to do some editing to get the 
# generated code to work. 
# It is completely harmless except for a few file existence checks at startup.
CFLAGS = -g -Wall -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER 

#CFLAGS = -O3 -Wall -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER -DGENERATE_WRAPPERS -DNDEBUG
#CFLAGS = -O3 -Wall -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER

# Use the options below for compiling on GCC4.1
# GNU C++ 4.1 and newer might need -ffriend-injection
#
#CFLAGS  =	-g -Wall -pedantic -DDO_UNIT_TESTS -ffriend-injection -DUSE_GENESIS_PARSER


# Libraries are defined below. For now we do not use threads.
SUBLIBS = 
#LIBS = 		-lm -lpthread
LIBS = 		-lm -lgsl -lgslcblas

# Here we decide if we want to use MPI and the parallel library
# Uncomment the line below if you do.
# The -DMPICH_IGNORE_CXX_SEEK flag is because of a bug in the
# MPI-2 standard. Hopefully it won't affect you, but if it does use
# the version of PARALLEL_FLAGS that defines this flag.
#PARALLEL_FLAGS = -DUSE_MPI
#PARALLEL_FLAGS = -DUSE_MPI -DMPICH_IGNORE_CXX_SEEK
#PARALLEL_DIR = parallel
#PARALLEL_LIB = parallel/parallel.o

# Depending on whether we compile with MPI, you may need to change the
# CXX compiler below
#
#CXX = mpicxx

#
# If you do use mpicxx, comment out the version below.
#
CXX = g++

LD = ld

SUBDIR = genesis_parser basecode shell element maindir biophysics kinetics builtins scheduling example utility $(PARALLEL_DIR)

OBJLIBS =	\
	basecode/basecode.o \
	utility/utility.o \
	maindir/maindir.o \
	genesis_parser/SLI.o \
	element/element.o \
	shell/shell.o \
	biophysics/biophysics.o \
	kinetics/kinetics.o \
	builtins/builtins.o \
	scheduling/scheduling.o \
	example/example.o \

export CFLAGS
export LD
export LIBS

moose: libs $(OBJLIBS) $(PARALLEL_LIB)
	$(CXX) $(CFLAGS) $(PARALLEL_FLAGS) $(OBJLIBS) $(PARALLEL_LIB) $(LIBS) -o moose
	@echo "Moose compilation finished"

libmoose.so: libs
	$(CXX) -G $(LIBS) -o libmoose.so
	@echo "Created dynamic library"

pymoose: CFLAGS += -fPIC
pymoose: SUBDIR += pymoose
pymoose: libs $(OBJLIBS) $(PARALLEL_LIB)
	$(MAKE) -C $@

libs:
	@(for i in $(SUBDIR); do echo cd $$i; cd $$i && $(MAKE) $(PARALLEL_FLAGS) ; cd ..; done)
	@echo "All Libs compiled"

mpp: preprocessor/*.cpp preprocessor/*.h
	@( rm -f mpp; cd preprocessor; make CXX="$(CXX)" CFLAGS="$(CFLAGS)"; ln mpp ..; cd ..)

default: moose mpp

clean:
	@(for i in $(SUBDIR) ; do echo cd $$i; cd $$i; $(MAKE) clean; cd ..; done)
	-rm -rf moose mpp core.* DOCS/html *.so *.py *.pyc
