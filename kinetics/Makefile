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
	Molecule.o	\
	Reaction.o	\
	Enzyme.o	\
	SparseMatrix.o	\
	KineticHub.o	\
	Stoich.o	\
	TestKinetics.o	\

HEADERS = \
	../basecode/header.h \
	../basecode/Conn.h \
	../basecode/Element.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
Molecule.o:	Molecule.h ../basecode/Ftype2.h
Reaction.o:	Reaction.h ../basecode/Ftype2.h
Enzyme.o:	Enzyme.h ../basecode/Ftype2.h
SparseMatrix.o:	SparseMatrix.h
KineticHub.o:	RateTerm.h Molecule.h Reaction.h Enzyme.h KineticHub.h
Stoich.o:	RateTerm.h Stoich.h SparseMatrix.h

.cpp.o:
	$(CXX) $(CFLAGS) -I../basecode $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
