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
	Stoich.o	\
	KinSparseMatrix.o	\
	ZombieMol.o \

	
	#KineticHub.o	\
	#testKsolve.o	\
	#Tab.o	\

HEADERS = \
	../basecode/header.h \


default: $(TARGET)

$(OBJ)	: $(HEADERS)
Stoich.o:	Stoich.h
KineticHub.o:	KineticHub.h
KinSparseMatrix.o:	KinSparseMatrix.h ../basecode/SparseMatrix.h
ZombieMol.o:	ZombieMol.h Stoich.h KinSparseMatrix.h ../basecode/SparseMatrix.h ../basecode/ElementValueFinfo.h
testKsolve.o:	Stoich.h KineticHub.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I.. -I../basecode -I../msg $< -c

$(TARGET):		$(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
