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
	Boundary.o	\
	ChemCompt.o	\
	ReadKkit.o	\
	MathFunc.o	\
	SumFunc.o	\
	testKinetics.o	\

HEADERS = \
	../basecode/header.h \


default: $(TARGET)

$(OBJ)	: $(HEADERS)
Mol.o:	Pool.h
BufMol.o:	Pool.h BufPool.h
FuncMol.o:	Pool.h FuncPool.h
Reac.o:	Reac.h
Enz.o:	Enz.h
MMenz.o:	MMenz.h 
Boundary.o:	Boundary.h
ChemCompt.o:	ChemCompt.h
ReadKkit.o:	ReadKkit.h Pool.h FuncPool.h
MathFunc.o:	MathFunc.h
SumFunc.o:	SumFunc.h
testKinetics.o:	ReadKkit.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I.. -I../basecode -I../msg $< -c

$(TARGET):		$(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
