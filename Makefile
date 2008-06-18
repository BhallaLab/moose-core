#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2003- 2006 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

# Linux compilation:
# We recommend two levels of compilation: either full debug, with gdb,
# unit tests and all the rest, or an optimized version for production
# simulations, without any unit tests or assertions. If you want some
# intermediate levels, edit the flags. Otherwise pick one of the two
# lines below:
#
# Debug mode:
CFLAGS = -g -Wall -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER

# Optimized mode:
# CFLAGS  =	-O3 -Wall -pedantic -DNDEBUG -DUSE_GENESIS_PARSER

##########################################################################
#
# MAC OS X compilation, Debug mode:
#CFLAGS  =	-g -Wall -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER -Wno-deprecated

# Optimized mode:
#CFLAGS  =	-Wall -pedantic -DUSE_GENESIS_PARSER -Wno-deprecated

##########################################################################
#
# Developer options (Don't try these unless you are writing new code!)
#
# For generating python interface:
# Do remember that you have to create a directory named "generated" 
# in the working directory of moose. Also you have to do some editing 
# to get the generated code to work. 
# Although this is verbose in its complaints, is completely harmless 
# except for a few file existence checks at startup.
#CFLAGS = -g -Wall -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER -DGENERATE_WRAPPERS -DUSE_MPI


# For parallel (MPI) version:
#CFLAGS = -g -Wall -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER -DUSE_MPI

# The -DMPICH_IGNORE_CXX_SEEK flag is because of a bug in the
# MPI-2 standard. Hopefully it won't affect you, but if it does use
# the version of CFLAGS that defines this flag:
#CFLAGS = -g -Wall -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER -DUSE_MPI -DMPICH_IGNORE_CXX_SEEK


#use this for readline library
#CFLAGS = -g -Wall -pedantic -DDO_UNIT_TESTS -DUSE_GENESIS_PARSER -DUSE_READLINE

# Use the options below for compiling on GCC4.1
# GNU C++ 4.1 and newer might need -ffriend-injection
#
#CFLAGS  =	-g -Wall -pedantic -DDO_UNIT_TESTS -ffriend-injection -DUSE_GENESIS_PARSER

##########################################################################
#
# Don't mess with stuff below!
#
##########################################################################


# Libraries are defined below. For now we do not use threads.
SUBLIBS = 
#LIBS = 		-lm -lpthread
LIBS = 		-lm 
#use this to use readline library
#LIBS = 		-lm -lreadline

# Here we automagically change compilers to deal with MPI.

ifneq (,$(findstring DUSE_MPI,$(CFLAGS)))
       CXX = mpicxx
       PARALLEL_DIR = parallel
       PARALLEL_LIB = parallel/parallel.o
else
       CXX = g++
endif

LD = ld

SUBDIR = genesis_parser basecode connections shell element maindir scheduling biophysics kinetics builtins $(PARALLEL_DIR) utility utility/randnum


OBJLIBS =	\
	basecode/basecode.o \
	connections/connections.o \
	maindir/maindir.o \
	genesis_parser/SLI.o \
	element/element.o \
	shell/shell.o \
	utility/utility.o \
	utility/randnum/randnum.o	\
	scheduling/scheduling.o \
	biophysics/biophysics.o \
	kinetics/kinetics.o \
	builtins/builtins.o \

# example/example.o 

export CFLAGS
export LD
export LIBS

moose: libs $(OBJLIBS) $(PARALLEL_LIB)
	$(CXX) $(CFLAGS) $(OBJLIBS) $(PARALLEL_LIB) $(LIBS) -o moose
	@echo "Moose compilation finished"

libmoose.so: libs
	$(CXX) -G $(LIBS) -o libmoose.so
	@echo "Created dynamic library"

pymoose: CFLAGS += -fPIC
pymoose: SUBDIR += pymoose
pymoose: libs $(OBJLIBS) $(PARALLEL_LIB)
	$(MAKE) -C $@

libs:
	@(for i in $(SUBDIR); do echo cd $$i; cd $$i && $(MAKE); cd ..; done)
	@echo "All Libs compiled"

mpp: preprocessor/*.cpp preprocessor/*.h
	@( rm -f mpp; cd preprocessor; make CXX="$(CXX)" CFLAGS="$(CFLAGS)"; ln mpp ..; cd ..)

default: moose mpp

clean:
	@(for i in $(SUBDIR) ; do echo cd $$i; cd $$i; $(MAKE) clean; cd ..; done)
	-rm -rf moose mpp core.* DOCS/html *.so *.py *.pyc
