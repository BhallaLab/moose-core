#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = ksolve.o

OBJ = \
	KinSparseMatrix.o	\
	KineticHub.o	\
	Stoich.o	\
	GssaStoich.o	\
	InterSolverFlux.o	\
	Kintegrator.o	\
	KineticManager.o	\
	TestKsolve.o	\

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \
	../kinetics/Molecule.h \
	../kinetics/Reaction.h \
	../kinetics/Enzyme.h \
	../randnum/randnum.h \


# The GSL library (GNU Scientific Library) provides a range of
# numerical functions, which allow us to use various advanced integration
# methods. The GslIntegrator accesses these. This library is licenced
# under the GPL, so only GPL-compliant uses of MOOSE will be able to
# use it. Should be fine for academic use.
# Optional: Comment out the following three items if not using GSL
ifeq ($(USE_GSL),1)
GSL_OBJ = GslIntegrator.o SteadyState.o StateScanner.o
#GSL_LIBS = -L/usr/lib -lgsl -lgslcblas
GSL_FLAGS = -DUSE_GSL
endif

default: $(TARGET)

$(OBJ)	: $(HEADERS)
KinSparseMatrix.o:	KinSparseMatrix.h
KineticHub.o:	RateTerm.h KineticHub.h
Stoich.o:	RateTerm.h Stoich.h KinSparseMatrix.h InterSolverFlux.h Makefile 
GssaStoich.o:	RateTerm.h Stoich.h KinSparseMatrix.h GssaStoich.h InterSolverFlux.h Makefile
InterSolverFlux.o:	InterSolverFlux.h
Kintegrator.o:	Kintegrator.h
KineticManager.o:	KineticManager.h ../kinetics/KinCompt.h
TestKsolve.o:	KineticManager.h RateTerm.h Stoich.h KinSparseMatrix.h KineticHub.h Makefile

ifeq ($(USE_GSL),1)
#Optional: Comment out the following two items if not using GSL.
$(GSL_OBJ) : $(HEADERS)
GslIntegrator.o:	GslIntegrator.h Stoich.h RateTerm.h KinSparseMatrix.h 
SteadyState.o:	SteadyState.h RateTerm.h KinSparseMatrix.h 
StateScanner.o:	StateScanner.h
endif # !USE_GSL

.cpp.o:
	$(CXX) $(CXXFLAGS) $(GSL_FLAGS) -I.. -I../basecode -I../kinetics -I../randnum $< -c

$(TARGET):		$(OBJ) $(GSL_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(GSL_OBJ) $(GSL_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
