#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU Lesser General Public License version 2.1
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = _rt.o

OBJ = \
	regressionTest.o	\
	rtTable.o	\
	rtReadKkit.o	\
	rtHHnetwork.o	\
	rtReacDiff.o	\
	rtMultiCompartment.o	\
	perfTestMarkovSolver.o \
	benchmarkTests.o	\

HEADERS = \
	../basecode/header.h \
	../shell/Shell.h

default: $(TARGET)

$(OBJ)	: $(HEADERS)
rtReadKkit.o:	../kinetics/ReadKkit.h ../shell/LoadModels.h
rReacDiff.o:	../kinetics/ReadCspace.h ../shell/LoadModels.h
rtMultiCompartment.o:	../kinetics/ReadKkit.h ../shell/LoadModels.h ../kinetics/PoolBase.h ../ksolve/SolverJunction.h ../ksolve/SolverBase.h ../ksolve/VoxelPools.h ../ksolve/RateTerm.h ../ksolve/StoichCore.h ../ksolve/OdeSystem.h ../ksolve/GslStoich.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I. -I../basecode -I../msg -I../shell $< -c

$(TARGET):		$(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
