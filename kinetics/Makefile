#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = kinetics.o

OBJ = \
	Pool.o	\
	BufPool.o	\
	FuncPool.o	\
	Reac.o	\
	Enz.o	\
	MMenz.o	\
	Species.o	\
	ReadKkit.o	\
	ReadCspace.o	\
	MathFunc.o	\
	SumFunc.o	\
	lookupSizeFromMesh.o	\
	testKinetics.o	\

HEADERS = \
	../basecode/header.h \


#	Boundary.o	\
#	ChemCompt.o	\

default: $(TARGET)

$(OBJ)	: $(HEADERS)
Pool.o:	Pool.h lookupSizeFromMesh.h
BufPool.o:	Pool.h BufPool.h lookupSizeFromMesh.h
FuncPool.o:	Pool.h FuncPool.h lookupSizeFromMesh.h
Reac.o:	Reac.h lookupSizeFromMesh.h
Enz.o:	Enz.h lookupSizeFromMesh.h
MMenz.o:	MMenz.h  lookupSizeFromMesh.h
Boundary.o:	Boundary.h
ChemCompt.o:	ChemCompt.h
Species.o:	Species.h
ReadKkit.o:	lookupSizeFromMesh.h ReadKkit.h Pool.h FuncPool.h ../shell/Shell.h 
ReadCspace.o:	lookupSizeFromMesh.h ReadCspace.h ../shell/Shell.h
MathFunc.o:	MathFunc.h
SumFunc.o:	SumFunc.h
lookupSizeFromMesh.o: lookupSizeFromMesh.h
testKinetics.o:	ReadKkit.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I.. -I../basecode -I../msg $< -c

$(TARGET):		$(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
