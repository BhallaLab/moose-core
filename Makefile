#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2003- 2005 Upinder S. Bhalla. and NCBS
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
#CFLAGS  =	-g -Wall -pedantic -DYYMALLOC -DYYFREE -DDO_UNIT_TESTS
#CFLAGS  =	-g -Wall -pedantic -DDO_UNIT_TESTS -DNO_OFFSETOF
#CFLAGS  =	-O3 -Wall -pedantic -DNO_OFFSETOF
#CFLAGS  =	-O3 -pg -Wall -pedantic -DNO_OFFSETOFF

# Use the options below for compiling on GCC4
# ANSI C++ and hence gcc4 have some strange error messages that emanate
# from offsetof, even when it does the right thing. The 
# -Wno-invalid-offsetof flag suppresses these silly warnings.
#  For Debian/Ubuntu 6.06, we need to add a few more compiler flags to
#  help it through the genesis parser, which is littered with ifdefs.
CFLAGS  =	-g -Wall -DYYMALLOC -DYYFREE -pedantic -DDO_UNIT_TESTS
# GNU C++ 4.1 and newer will need -ffriend-injection
#CFLAGS  =	-g -Wall -ffriend-injection -Weffc++ -DYYMALLOC -DYYFREE -pedantic -DDO_UNIT_TESTS
#CFLAGS  =	-g -Wall -pedantic -DDO_UNIT_TESTS -Wno-invalid-offsetof
#CFLAGS  =	-O3 -Wall -pedantic -Wno-invalid-offsetof


# Libraries are defined below. For now we do not use threads.
SUBLIBS = 
#LIBS = 		-lm -lpthread
LIBS = 		-lm

#
CXX = g++
LD = ld

SUBDIR = basecode maindir randnum builtins genesis_parser scheduling kinetics biophysics textio gui

OBJLIBS =	\
	basecode/basecode.o \
	scheduling/scheduling.o \
	randnum/randnum.o \
	maindir/maindir.o \
	genesis_parser/SLI.o \
	builtins/builtins.o \
	textio/textio.o \
	kinetics/kinetics.o \
	biophysics/biophysics.o \
	gui/gui.o \


libs:
	@(for i in $(SUBDIR); do echo cd $$i; cd $$i; make CXX="$(CXX)" CFLAGS="$(CFLAGS)" LD="$(LD)" LIBS="$(SUBLIBS)"; cd ..; done)
	@echo "All Libs compiled"

moose: libs $(OBJLIBS)
	$(CXX) $(CFLAGS) $(OBJLIBS) $(LIBS) -o moose
	@echo "Moose compilation finished"

mpp: preprocessor/*.cpp preprocessor/*.h
	@( rm -f mpp; cd preprocessor; make CXX="$(CXX)" CFLAGS="$(CFLAGS)"; ln mpp ..; cd ..)

default: moose mpp

clean:
	@(for i in $(SUBDIR) ; do echo cd $$i; cd $$i; make clean; cd ..; done)
	-rm -rf moose mpp core.* 
