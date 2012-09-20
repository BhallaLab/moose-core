#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = _kinetics.o

OBJ = \
	PoolBase.o	\
	Pool.o	\
	BufPool.o	\
	FuncPool.o	\
	ReacBase.o	\
	Reac.o	\
	EnzBase.o	\
	CplxEnzBase.o	\
	Enz.o	\
	MMenz.o	\
	Species.o	\
	ReadKkit.o	\
	WriteKkit.o	\
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
Pool.o:	PoolBase.h Pool.h lookupSizeFromMesh.h
BufPool.o:	PoolBase.h Pool.h BufPool.h lookupSizeFromMesh.h
FuncPool.o:	PoolBase.h Pool.h FuncPool.h lookupSizeFromMesh.h
ReacBase.o:	ReacBase.h lookupSizeFromMesh.h
Reac.o:	ReacBase.h Reac.h lookupSizeFromMesh.h
EnzBase.o:	EnzBase.h lookupSizeFromMesh.h
CplxEnzBase.o:	EnzBase.h CplxEnzBase.h lookupSizeFromMesh.h
Enz.o:	EnzBase.h CplxEnzBase.h Enz.h lookupSizeFromMesh.h
MMenz.o:	EnzBase.h MMenz.h  lookupSizeFromMesh.h
Boundary.o:	Boundary.h
ChemCompt.o:	ChemCompt.h
Species.o:	Species.h
ReadKkit.o:	lookupSizeFromMesh.h ReadKkit.h PoolBase.h Pool.h FuncPool.h ../shell/Shell.h 
WriteKkit.o:	lookupSizeFromMesh.h PoolBase.h
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
