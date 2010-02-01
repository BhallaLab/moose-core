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

TARGET=pymoose
CXXFLAGS+=-I/usr/local/include -I.. -I../basecode -I../connections -I../maindir -I../genesis_parser  -I../shell -I../element -I../scheduling -I../biophysics -I../hsolve -I../kinetics -I../ksolve -I../builtins -I../utility -I../randnum -I../signeur -I../device
LIBS += -L/usr/local/lib -lpython2.6
ifdef USE_NUMPY
CXXFLAGS+=-I/usr/lib/python2.6/site-packages/numpy/core/include -DUSE_NUMPY
SWIGOPTS=-DUSE_NUMPY
endif
ifdef USE_GL
CXXFLAGS+=-I../gl/src -DUSE_GL
SWIGOPTS=-DUSE_GL
endif
# The library location and flags for creating dll in Mac is different from Linux
# Darwin may not have the OSTYPE environment variable defined in bash. Set OSTYPE=Darwin if required
ifeq ($(OSTYPE),Darwin)
	CXXFLAGS+=-bundle -fno-common /usr/bin/python -flat_namespace -undefined suppress -I/Library/Frameworks/Python.framework/Versions/2.5/include/python2.5
	LDFLAGS+=-bundle -fno-common -L/Library/Frameworks/Python.framework/Versions/2.5/lib/python2.5/config/
else # this is for unix like systems - change /usr/include/Python2.5 to the directory where Python.h is located
	CXXFLAGS +=-I/usr/include/python2.6
	LDFLAGS+=-shared
endif

OBJFILES := ../basecode/basecode.o \
	../element/element.o \
	../shell/shell.o \
	../builtins/builtins.o \
	../biophysics/biophysics.o \
	../scheduling/scheduling.o \
	../maindir/init.o	\
	../maindir/initCinfos.o	\
	../connections/connections.o \
	../utility/utility.o	\
	../randnum/randnum.o	\
	../maindir/init.o	\
	../connections/connections.o \
	../genesis_parser/SLI.o	\
	../kinetics/kinetics.o	\
	../hsolve/hsolve.o	\
	../ksolve/ksolve.o	\
	../device/device.o \
	../signeur/signeur.o	\
# 	../robots/robots.o	\

ifeq ($(USE_SBML),1)
	OBJFILES := $(OBJFILES) ../sbml_IO/sbml_IO.o
endif

ifeq ($(USE_NEUROML),1)
	LIBS+= -L../external/lib/
endif

ifeq ($(USE_GL),1)
	OBJFILES := $(OBJFILES) ../gl/src/GLcell.o ../gl/src/GLview.o ../gl/src/GLshape.o
endif

OBJ = PyMooseBase.o \
	PyMooseContext.o \
	Class.o	\
	Cell.o	\
	Compartment.o \
	Neutral.o \
	PyMooseUtil.o \
	HHChannel.o \
	HHGate.o \
	Mg_block.o	\
	Interpol.o \
	CaConc.o \
	SpikeGen.o \
	PulseGen.o	\
	RandomSpike.o	\
	SynChan.o \
	BinSynchan.o	\
	StochSynchan.o	\
	Table.o \
	Nernst.o \
	Tick.o \
	ClockJob.o \
	Enzyme.o \
	KineticHub.o \
	Kintegrator.o \
	KineticManager.o \
	Molecule.o \
	Reaction.o \
	Stoich.o \
	NeuroScan.o \
	HSolve.o \
	RandGenerator.o	\
	BinomialRng.o	\
	ExponentialRng.o	\
	GammaRng.o	\
	NormalRng.o	\
	PoissonRng.o	\
	UniformRng.o	\
	KinCompt.o \
	Panel.o \
	DiskPanel.o \
	HemispherePanel.o \
	RectPanel.o \
	CylPanel.o \
	SpherePanel.o \
	TriPanel.o \
	Surface.o \
	Geometry.o	\
	MathFunc.o	\
	AscFile.o	\
	DifShell.o	\
	GssaStoich.o	\
	TauPump.o	\
	GLcell.o	\
	GLview.o	\
	TimeTable.o	\
	PIDController.o \
	RC.o	\
	DiffAmp.o	\
	GHK.o	\
	IntFire.o	\
	IzhikevichNrn.o	\
	HHChannel2D.o	\
	HHGate2D.o	\
	SteadyState.o	\
	InterSolverFlux.o	\
	Adaptor.o	\
 	SigNeur.o	\
	Leakage.o	\


	
HEADERS=PyMooseUtil.h \
	PyMooseContext.h \
	PyMooseBase.h \
	Compartment.h \
	Neutral.h \
	Class.h	\
	Cell.h	\
	Leakage.h	\
	HHChannel.h \
	HHGate.h \
	Mg_block.h	\
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
	KineticManager.h	\
	Molecule.h \
	Reaction.h \
	Stoich.h  \
	NeuroScan.h \
	HSolve.h	\
	RandGenerator.h	\
	BinomialRng.h	\
	ExponentialRng.h	\
	GammaRng.h	\
	NormalRng.h	\
	PoissonRng.h	\
	UniformRng.h	\
	KinCompt.h	\
	Panel.h	\
	DiskPanel.h	\
	CylPanel.h	\
	HemispherePanel.h	\
	SpherePanel.h	\
	TriPanel.h	\
	RectPanel.h	\
	Surface.h	\
	Geometry.h	\
	MathFunc.h	\
	AscFile.h	\
	DifShell.h	\
	GssaStoich.h	\
	TauPump.h	\
	GLcell.h	\
	GLview.h	\
	TimeTable.h	\
	PIDController.h \
	RC.h	\
	DiffAmp.h	\
	GHK.h	\
	IntFire.h	\
	IzhikevichNrn.h	\
	HHChannel2D.h	\
	HHGate2D.h	\
	SteadyState.h	\
	InterSolverFlux.h	\
	../randnum/randnum.h	\
	../randnum/Probability.h	\
	../randnum/Binomial.h	\
	../randnum/Exponential.h	\
	../randnum/Gamma.h	\
	../randnum/Normal.h	\
	../randnum/Poisson.h	\
	../maindir/init.h		\
  	Adaptor.h	\
  	SigNeur.h	\


#TEST_MAIN = TestPyMoose.o

#testPyMoose: $(OBJ) $(OBJFILES) $(HEADERS) $(TEST_MAIN)
#	$(CXX) $(CXXFLAGS) -o $@ $^


default: $(TARGET)

.PHONY: pymoose
.PHONY: clean

pymoose: _moose.so moose.py #testPyMoose 

ifeq ($(OSTYPE),Darwin)
_moose.so: moose_wrap.o $(OBJFILES) $(OBJ) 
	$(CXX) $(LDFLAGS) -o $@  $(LIBS) $^ 
endif

ifneq ($(OSTYPE),Darwin)
_moose.so: moose_wrap.o $(OBJ) $(OBJFILES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)
endif

moose_wrap.o: moose_wrap.cxx
	$(CXX) $(CXXFLAGS) -c $^ -o $@

moose.py moose_wrap.cxx: moose.i $(HEADERS)
	swig -modern -c++ -python -threads $(SWIGOPTS) moose.i

# This is not required - gnumake does it by implicit rule
# .cpp.o:
# 	$(CXX) $(CXXFLAGS) $< -c

clean:
	-rm -f *.o *.so test_swig *.pyc
#	-rm -f *.o moose_wrap.cxx *.so moose.py test_swig *.pyc
