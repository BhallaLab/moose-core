#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU Lesser General Public License version 2.1
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = _ksolve.o

OBJ = \
	KinSparseMatrix.o	\
	ZombiePool.o \
	VoxelPools.o \
	RateTerm.o \
	Stoich.o \
	Ksolve.o \

HEADERS = \
	../basecode/header.h \
	../basecode/SparseMatrix.h \
	../basecode/ElementValueFinfo.h \
	RateTerm.h \
	KinSparseMatrix.h \
	../kinetics/Pool.h \
	../kinetics/lookupVolumeFromMesh.h \


# The GSL library (GNU Scientific Library) provides a range of
# numerical functions, which allow us to use various advanced integration
# methods. The GslIntegrator accesses these. This library is licenced
# under the GPL, so only GPL-compliant uses of MOOSE will be able to
# use it. Should be fine for academic use.
# Optional: Comment out the following three items if not using GSL
ifeq ($(USE_GSL),1)
GSL_OBJ = \
#	GslIntegrator.o

#GSL_LIBS = -L/usr/lib -lgsl -lgslcblas
#GSL_FLAGS = -DUSE_GSL
endif

default: $(TARGET)

$(OBJ)	: $(HEADERS)
KinSparseMatrix.o:	KinSparseMatrix.h ../basecode/SparseMatrix.h
ZombiePool.o:	../kinetics/PoolBase.h ZombiePool.h
VoxelPools.o:	VoxelPools.h RateTerm.h Stoich.h
RateTerm.o:		RateTerm.h
Stoich.o:		RateTerm.h ../kinetics/FuncTerm.h ../kinetics/SumTotalTerm.h Stoich.h ../kinetics/PoolBase.h ../kinetics/ReacBase.h ../kinetics/EnzBase.h ../kinetics/CplxEnzBase.h ../kinetics/FuncBase.h
Ksolve.o:		RateTerm.h Stoich.h Ksolve.h


#KineticHub.o:	KineticHub.h

ifeq ($(USE_GSL),1)
#Optional: Comment out the following two items if not using GSL.
$(GSL_OBJ) : $(HEADERS)
#GslIntegrator.o:	GslIntegrator.h Stoich.h RateTerm.h KinSparseMatrix.h ../basecode/SparseMatrix.h ../shell/Shell.h
#SteadyState.o:	SteadyState.h RateTerm.h KinSparseMatrix.h 
#StateScanner.o:	StateScanner.h
endif # !USE_GSL

.cpp.o:
	$(CXX) $(CXXFLAGS) $(GSL_FLAGS) $(SMOLDYN_FLAGS) -I.. -I../basecode -I../msg -I../kinetics -I../mesh $< -c

$(TARGET):		$(OBJ) $(GSL_OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(GSL_OBJ) $(GSL_LIBS) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
