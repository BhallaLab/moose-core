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
	Mol.o	\
	Reac.o	\
	Enz.o	\
	MMenz.o	\
	Boundary.o	\
	ChemCompt.o	\
	ReadKkit.o	\
	testKinetics.o	\

	#Tab.o	\

HEADERS = \
	../basecode/header.h \


default: $(TARGET)

$(OBJ)	: $(HEADERS)
Mol.o:	Mol.h ../basecode/Data.h
Reac.o:	Reac.h ../basecode/Data.h
Enz.o:	Enz.h ../basecode/Data.h
MMenz.o:	MMenz.h ../basecode/Data.h
Boundary.o:	Boundary.h ../basecode/Data.h
ChemCompt.o:	ChemCompt.h ../basecode/Data.h
ReadKkit.o:	ReadKkit.h
Tab.o:	Tab.h ../basecode/Data.h
testKinetics.o:	ReadKkit.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I.. -I../basecode -I../msg $< -c

$(TARGET):		$(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
