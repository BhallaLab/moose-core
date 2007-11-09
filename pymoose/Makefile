############################################################################
# To avoid portability issues, we are avoiding setting the
# include directories of the third party libraries in the 
# makefile. 
# To build the project using make, you should set the proper 
# environment variables according to the location of the third 
# party softwares.
# These environment variables are:
# CPLUS_INCLUDE_PATH
# This should have python include directory in it, for linux 
# it may be: /usr/include/python{VERSION} 
#	/usr/local/include/python{VERSION}
# For Apple's Darwin ( Mac), it might be:
# /Library/Frameworks/Python.framework/Versions/2.5/include/python{VERSION}
#
# Also, you should pass the non-essential flags for compilation in the 
# command line or as environment variables.
# A sample build session could be:
# export CPLUS_INCLUDE_PATH=/usr/local/include/python2.5
# export CXXFLAGS=-g -Wall
# export LDFLAGS=-lm -lpython
# make pymoose
#
############################################################################

TARGET = pymoose
CXXFLAGS +=-DDO_UNIT_TESTS -fPIC
DYNAMIC_FLAGS=-shared 
# The library location and flags for creating dll in Mac is different from Linux
SYSTEM=$(shell (uname))
ifeq ($(SYSTEM),Darwin)
	DYNAMIC_FLAGS=-bundle -flat_namespace -undefined suppress 
endif

INCLUDES=-I/usr/include/python2.4 -I..

OBJFILES = ../basecode/basecode.o \
	../element/element.o \
	../shell/shell.o \
	../builtins/builtins.o \
	../biophysics/biophysics.o \
	../scheduling/scheduling.o \
	../utility/utility.o	\
	../utility/randnum/randnum.o	\


OBJ = PyMooseBase.o \
	PyMooseContext.o \
	Class.o	\
	Cell.o	\
	Compartment.o \
	Neutral.o \
	PyMooseUtil.o \
	HHChannel.o \
	HHGate.o \
	Interpol.o \
	CaConc.o \
	SpikeGen.o \
	SynChan.o \
	Table.o \
	Nernst.o \
	Tick.o \
	ClockJob.o \
	Enzyme.o \
	KineticHub.o \
	Kintegrator.o \
	Molecule.o \
	Reaction.o \
	Stoich.o \
	HSolve.o	\
	RandGenerator.o	\
	BinomialRng.o	\
	ExponentialRng.o	\
	GammaRng.o	\
	NormalRng.o	\
	PoissonRng.o	\


HEADERS = PyMooseUtil.h \
	PyMooseContext.h \
	PyMooseBase.h \
	Compartment.h \
	Neutral.h \
	Class.h	\
	Cell.h	\
	HHChannel.h \
	HHGate.h \
	Interpol.h \
	CaConc.h \
	SpikeGen.h \
	SynChan.h \
	Table.h \
	Nernst.h \
	Tick.h \
	ClockJob.h \
	TableIterator.h \
	Enzyme.h \
	KineticHub.h \
	Kintegrator.h \
	Molecule.h \
	Reaction.h \
	Stoich.h  \
	HSolve.h	\
	RandGenerator.h	\
	BinomialRng.h	\
	ExponentialRng.h	\
	GammaRng.h	\
	NormalRng.h	\
	PoissonRng.h	\
	../utility/randnum/randnum.h	\
	../utility/randnum/Probability.h	\
	../utility/randnum/Binomial.h	\
	../utility/randnum/Exponential.h	\
	../utility/randnum/Gamma.h	\
	../utility/randnum/Normal.h	\
	../utility/randnum/Poisson.h	\


# TEST_MAIN = SquidTest.cpp
# TEST_MAIN = SwigTest.cpp

#test_swig: $(OBJFILES) $(HEADERS) $(TEST_MAIN)
#	$(CXX) $(CXXFLAGS) -o test_swig $(TEST_MAIN) $(OBJFILES)  


default: $(TARGET)

.PHONY: pymoose
.PHONY: clean

pymoose: _moose.so moose.py 

_moose.so: moose_wrap.o $(OBJ) $(OBJFILES)
	$(CXX) $(CXXFLAGS) $(DYNAMIC_FLAGS) -o $@ $^ $(LDFLAGS)

moose_wrap.o: moose_wrap.cxx
	$(CXX) $(CXXFLAGS) -c $^ -o $@ $(INCLUDES)

moose.py moose_wrap.cxx: moose.i $(HEADERS)
	swig -modern -c++ -python moose.i

# This is not required - gnumake does it by implicit rule
# .cpp.o:
# 	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -c

clean:
	-rm -f *.o moose_wrap.cxx *.so moose.py test_swig *.pyc


