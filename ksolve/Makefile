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
	Port.o	\
	Stoich.o	\
	KinSparseMatrix.o	\
	ZombieMol.o \
	ZombieBufMol.o \
	ZombieFuncMol.o \
	ZombieReac.o \
	ZombieEnz.o \
	ZombieMMenz.o \
	ZombieSumFunc.o \
	RateTerm.o \
	FuncTerm.o \
	testKsolve.o	\

	
	#KineticHub.o	\
	#Tab.o	\

HEADERS = \
	../basecode/header.h \
	../basecode/SparseMatrix.h \
	../basecode/ElementValueFinfo.h \
	Port.h \
	Stoich.h \
	RateTerm.h \
	FuncTerm.h \
	KinSparseMatrix.h \
	../kinetics/Mol.h \


# The GSL library (GNU Scientific Library) provides a range of
# numerical functions, which allow us to use various advanced integration
# methods. The GslIntegrator accesses these. This library is licenced
# under the GPL, so only GPL-compliant uses of MOOSE will be able to
# use it. Should be fine for academic use.
# Optional: Comment out the following three items if not using GSL
ifeq ($(USE_GSL),1)
GSL_OBJ = \
	GslIntegrator.o

#GSL_LIBS = -L/usr/lib -lgsl -lgslcblas
#GSL_FLAGS = -DUSE_GSL
endif

default: $(TARGET)

$(OBJ)	: $(HEADERS)
Stoich.o:	ZombieMol.h ZombieReac.h ZombieBufMol.h ZombieEnz.h ZombieMMenz.h ../kinetics/Mol.h ../kinetics/Reac.h ../kinetics/BufMol.h ../kinetics/FuncMol.h ../kinetics/Enz.h ../kinetics/MMenz.h ../kinetics/Boundary.h ../kinetics/ChemCompt.h
KinSparseMatrix.o:	KinSparseMatrix.h ../basecode/SparseMatrix.h
ZombieMol.o:	ZombieMol.h
ZombieBufMol.o:	ZombieMol.h ZombieBufMol.h
ZombieFuncMol.o:	ZombieMol.h ZombieFuncMol.h ZombieSumFunc.h
ZombieReac.o:	../kinetics/Reac.h ZombieReac.h
ZombieEnz.o:	ZombieEnz.h
ZombieMMenz.o:	ZombieMMenz.h
ZombieSumFunc.o:	../kinetics/SumFunc.h ZombieSumFunc.h
testKsolve.o:	../kinetics/ReadKkit.h ../shell/Shell.h
RateTerm.o:		RateTerm.h
FuncTerm.o:		FuncTerm.h

#KineticHub.o:	KineticHub.h

ifeq ($(USE_GSL),1)
#Optional: Comment out the following two items if not using GSL.
$(GSL_OBJ) : $(HEADERS)
GslIntegrator.o:	GslIntegrator.h Stoich.h RateTerm.h KinSparseMatrix.h ../basecode/SparseMatrix.h
#SteadyState.o:	SteadyState.h RateTerm.h KinSparseMatrix.h 
#StateScanner.o:	StateScanner.h
endif # !USE_GSL

.cpp.o:
	$(CXX) $(CXXFLAGS) $(GSL_FLAGS) $(SMOLDYN_FLAGS) -I.. -I../basecode -I../msg -I../kinetics $< -c

$(TARGET):		$(OBJ) $(GSL_OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(GSL_OBJ) $(GSL_LIBS) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
