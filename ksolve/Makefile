#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = _ksolve.o

OBJ = \
	Port.o	\
	Stoich.o	\
	StoichCore.o	\
	StoichPools.o	\
	KinSparseMatrix.o	\
	ZombiePool.o \
	ZPool.o \
	ZombieBufPool.o \
	ZBufPool.o \
	ZombieFuncPool.o \
	ZFuncPool.o \
	ZombieReac.o \
	ZReac.o \
	ZombieEnz.o \
	ZEnz.o \
	ZombieMMenz.o \
	ZMMenz.o \
	ZombieSumFunc.o \
	RateTerm.o \
	GssaStoich.o	\
	GslStoich.o	\
	EstimateDt.o	\
	testKsolve.o	\

	
	#KineticHub.o	\
	#Tab.o	\

HEADERS = \
	../basecode/header.h \
	../basecode/SparseMatrix.h \
	../basecode/ElementValueFinfo.h \
	Port.h \
	Stoich.h \
	StoichCore.h \
	StoichPools.h \
	RateTerm.h \
	KinSparseMatrix.h \
	../kinetics/Pool.h \
	../kinetics/lookupSizeFromMesh.h \


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
Stoich.o:	../kinetics/PoolBase.h ZombiePool.h ../kinetics/ReacBase.h ZombieReac.h ZombieBufPool.h ../kinetics/EnzBase.h ZombieEnz.h ZombieMMenz.h ../kinetics/Pool.h ../kinetics/Reac.h ../kinetics/BufPool.h ../kinetics/FuncPool.h ../kinetics/Enz.h ../kinetics/MMenz.h ../mesh/Boundary.h ../mesh/ChemMesh.h ../mesh/Stencil.h
StoichCore.o:	../kinetics/PoolBase.h ZombiePool.h ../kinetics/ReacBase.h ZombieReac.h ZombieBufPool.h ../kinetics/EnzBase.h ZombieEnz.h ZombieMMenz.h ../kinetics/Pool.h ../kinetics/Reac.h ../kinetics/BufPool.h ../kinetics/FuncPool.h ../kinetics/Enz.h ../kinetics/MMenz.h ../mesh/Boundary.h ../mesh/ChemMesh.h ../mesh/Stencil.h
StoichPools.o: StoichPools.h
GssaStoich.o:	GssaStoich.h ../kinetics/PoolBase.h ZombiePool.h ../kinetics/ReacBase.h ZombieReac.h ZombieBufPool.h ../kinetics/EnzBase.h ZombieEnz.h ZombieMMenz.h ../kinetics/Pool.h ../kinetics/Reac.h ../kinetics/BufPool.h ../kinetics/FuncPool.h ../kinetics/Enz.h ../kinetics/MMenz.h ../mesh/Boundary.h ../mesh/ChemMesh.h ../randnum/randnum.h
GslStoich.o:	GslStoich.h
KinSparseMatrix.o:	KinSparseMatrix.h ../basecode/SparseMatrix.h
ZombiePool.o:	../kinetics/PoolBase.h ZombiePool.h
ZPool.o:	../kinetics/PoolBase.h ZPool.h
ZombieBufPool.o:	../kinetics/PoolBase.h ZombiePool.h ZombieBufPool.h
ZBufPool.o:	../kinetics/PoolBase.h ZPool.h ZBufPool.h
ZombieFuncPool.o:	../kinetics/PoolBase.h ZombiePool.h ZombieFuncPool.h ZombieSumFunc.h
ZFuncPool.o:	../kinetics/PoolBase.h ZPool.h ZFuncPool.h
ZombieReac.o:	../kinetics/ReacBase.h ../kinetics/Reac.h ZombieReac.h
ZReac.o:	../kinetics/ReacBase.h ZReac.h
ZombieEnz.o:	ZombieEnz.h
ZEnz.o:	ZEnz.h
ZombieMMenz.o:	ZombieMMenz.h
ZMMenz.o:	ZMMenz.h
ZombieSumFunc.o:	../kinetics/SumFunc.h ZombieSumFunc.h
testKsolve.o:	../kinetics/ReadKkit.h ../shell/Shell.h
RateTerm.o:		RateTerm.h

#KineticHub.o:	KineticHub.h

ifeq ($(USE_GSL),1)
#Optional: Comment out the following two items if not using GSL.
$(GSL_OBJ) : $(HEADERS)
GslIntegrator.o:	GslIntegrator.h Stoich.h RateTerm.h KinSparseMatrix.h ../basecode/SparseMatrix.h ../shell/Shell.h
#SteadyState.o:	SteadyState.h RateTerm.h KinSparseMatrix.h 
#StateScanner.o:	StateScanner.h
endif # !USE_GSL

.cpp.o:
	$(CXX) $(CXXFLAGS) $(GSL_FLAGS) $(SMOLDYN_FLAGS) -I.. -I../basecode -I../msg -I../kinetics -I../mesh $< -c

$(TARGET):		$(OBJ) $(GSL_OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(GSL_OBJ) $(GSL_LIBS) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
