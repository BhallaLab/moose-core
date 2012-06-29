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

TARGET=pymoose.o
CXXFLAGS+= -fpermissive -fno-strict-aliasing -I/usr/local/include -I.. -I../basecode -I../connections -I../maindir -I../genesis_parser  -I../shell -I../element -I../scheduling -I../biophysics -I../hsolve -I../kinetics -I../ksolve -I../builtins -I../utility -I../randnum -I../robots -I../device

ifeq ($(USE_NEUROML),1)
CXXFLAGS+=-I../neuroML_IO -I../external/neuroML_src -DUSE_NEUROML
SWIGOPTS+=-DUSE_NEUROML
endif

# The library location and flags for creating dll in Mac is different from Linux
# Darwin may not have the OSTYPE environment variable defined in bash. Set OSTYPE=Darwin if required
ifeq ($(OSTYPE),Darwin)
	CXXFLAGS+=-bundle -fno-common /usr/bin/python -flat_namespace -undefined suppress -I/Library/Frameworks/Python.framework/Versions/2.6/include/python2.6
	export LDFLAGS+=-bundle -fno-common -L/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/config/
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
	Interpol2D.o \
	CaConc.o \
	SpikeGen.o \
	StochSpikeGen.o \
	Efield.o	\
	PulseGen.o	\
	RandomSpike.o	\
	SynChan.o \
	NMDAChan.o \
	BinSynchan.o	\
	StochSynchan.o	\
	STPSynChan.o	\
	STPNMDAChan.o	\
	KinSynChan.o \
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
	GslIntegrator.o	\
	SteadyState.o	\
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
	Interpol2D.h \
	CaConc.h \
	SpikeGen.h \
	StochSpikeGen.h \
	RandomSpike.h	\
	Efield.h	\
	PulseGen.h	\
	SynChan.h \
	NMDAChan.h \
	BinSynchan.h	\
	StochSynchan.h	\
	STPSynChan.h	\
	STPNMDAChan.h	\
	KinSynChan.h \
	Table.h \
	Nernst.h \
	Tick.h \
	ClockJob.h \
	TableIterator.h \
	Enzyme.h \
	KineticHub.h \
	Kintegrator.h \
	GslIntegrator.h	\
	SteadyState.h	\
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



default: $(TARGET)


.PHONY: clean

$(TARGET): moose.py moose_wrap.o $(OBJ) $(HEADERS)
	$(LD)  -r -o $(TARGET) $(OBJ) moose_wrap.o

moose_wrap.o: moose_wrap.cxx
	$(CXX) $(CXXFLAGS) -c $^ -o $@

moose.py moose_wrap.cxx: moose.i $(HEADERS) $(OBJ)
	swig -modern -c++ -python -threads $(SWIGOPTS) moose.i

# This is not required - gnumake does it by implicit rule
# .cpp.o:
# 	$(CXX) $(CXXFLAGS) $< -c

clean:
	-rm -f *.o moose_wrap.cxx *.so moose.py test_swig *.pyc
