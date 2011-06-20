#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = mesh.o

OBJ = \
	ChemMesh.o	\
	MeshEntry.o	\
	CylMesh.o	\
	CubeMesh.o	\
	testMesh.o	\

HEADERS = \
	../basecode/header.h \


default: $(TARGET)

$(OBJ)	: $(HEADERS)
MeshEntry.o:	MeshEntry.h
ChemMesh.o:	ChemMesh.h MeshEntry.h
CylMesh.o:	ChemMesh.h CylMesh.h MeshEntry.h
CubeMesh.o:	ChemMesh.h CubeMesh.h MeshEntry.h
testMesh.o:	MeshEntry.h ChemMesh.h CylMesh.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I.. -I../basecode $< -c

$(TARGET):		$(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
