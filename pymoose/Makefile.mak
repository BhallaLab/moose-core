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
DYNAMIC_FLAGS=-shared
# The library location and flags for creating dll in Mac is different from Linux
#SYSTEM=$(shell (uname))
#ifeq ($(SYSTEM),Darwin)
#	DYNAMIC_FLAGS=-bundle -flat_namespace -undefined suppress
#endif

LOCAL_DIR := pymoose


LOCAL_SRCS := \
    PyMooseBase.cpp \
	PyMooseContext.cpp \
	Class.cpp	\
	Cell.cpp	\
	Compartment.cpp \
	Neutral.cpp \
	PyMooseUtil.cpp \
	HHChannel.cpp \
	HHGate.cpp \
	Interpol.cpp \
	CaConc.cpp \
	SpikeGen.cpp \
	PulseGen.cpp	\
	RandomSpike.cpp	\
	SynChan.cpp \
	BinSynchan.cpp	\
	StochSynchan.cpp	\
	Table.cpp \
	Nernst.cpp \
	Tick.cpp \
	ClockJob.cpp \
	Enzyme.cpp \
	KineticHub.cpp \
	Kintegrator.cpp \
	Molecule.cpp \
	Reaction.cpp \
	Stoich.cpp \
	HSolve.cpp	\
	RandGenerator.cpp	\
	BinomialRng.cpp	\
	ExponentialRng.cpp	\
	GammaRng.cpp	\
	NormalRng.cpp	\
	PoissonRng.cpp	\
	UniformRng.cpp	\


LOCAL_HEADERS = PyMooseUtil.h \
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
	RandomSpike.h	\
	PulseGen.h	\
	SynChan.h \
	BinSynchan.h	\
	StochSynchan.h	\
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
	UniformRng.h	\


SOURCES += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_SRCS))

HEADERS += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_HEADERS))


INCLUDES= -I..
#-I/usr/include/python2.5
EXT_OBJ = \
    ..$(/)basecode$(/)basecode.o \
	..$(/)element$(/)element.o \
	..$(/)shell$(/)shell.o \
	..$(/)builtins$(/)builtins.o \
	..$(/)biophysics$(/)biophysics.o \
	..$(/)scheduling$(/)scheduling.o \
	..$(/)utility$(/)utility.o	\
	..$(/)utility$(/)randnum$(/)randnum.o	\
	..$(/)maindir$(/)init.o	\
	..$(/)connections$(/)connections.o


EXT_HEADERS = \
    ..$(/)utility$(/)randnum$(/)randnum.h	\
	..$(/)utility$(/)randnum$(/)Probability.h	\
	..$(/)utility$(/)randnum$(/)Binomial.h	\
	..$(/)utility$(/)randnum$(/)Exponential.h	\
	..$(/)utility$(/)randnum$(/)Gamma.h	\
	..$(/)utility$(/)randnum$(/)Normal.h	\
	..$(/)utility$(/)randnum$(/)Poisson.h	\
	..$(/)maindir$(/)init.h		\

#TEST_MAIN = TestPyMoose.o

#testPyMoose: $(OBJ) $(OBJFILES) $(HEADERS) $(TEST_MAIN)
#	$(CXX) $(CXXFLAGS) -o $@ $^


default: $(TARGET)

.PHONY: pymoose
.PHONY: clean

pymoose: $(LOCAL_DIR)$/_moose.so $(LOCAL_DIR)$(/)moose.py

$(LOCAL_DIR)$(/)_moose.so: $(LOCAL_DIR)$(/)moose_wrap.o $(OBJ) $(OBJFILES)
	$(CXX) $(CXXFLAGS) $(DYNAMIC_FLAGS) -o $@ $^ $(LDFLAGS)

$(LOCAL_DIR)$(/)moose_wrap.o: $(LOCAL_DIR)$(/)moose_wrap.cxx
	$(CXX) $(CXXFLAGS) -c $^ -o $@ $(INCLUDES)

$(LOCAL_DIR)$(/)moose.py $(LOCAL_DIR)$(/)moose_wrap.cxx: $(LOCAL_DIR)$(/)moose.i $(HEADERS)
	swig -modern -c++ -python -threads $(LOCAL_DIR)$(/)moose.i




