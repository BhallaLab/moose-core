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
	Kintegrator.o	\
	MathFunc.o	\
	TestKinetics.o	\

HEADERS = \
	../basecode/header.h \
	../basecode/Conn.h \
	../basecode/Element.h \


# The GSL library (GNU Scientific Library) provides a range of
# numerical functions, which allow us to use various advanced integration
# methods. The GslIntegrator accesses these. This library is licenced
# under the GPL, so only GPL-compliant uses of MOOSE will be able to
# use it. Should be fine for academic use.
# Optional: Comment out the following three items if not using GSL
GSL_OBJ = GslIntegrator.o
GSL_LIBS = -lgsl -lgslcblas
GSL_FLAGS = -DUSE_GSL


default: $(TARGET)

$(OBJ)	: $(HEADERS)
Molecule.o:	Molecule.h ../basecode/Ftype2.h
Reaction.o:	Reaction.h ../basecode/Ftype2.h
Enzyme.o:	Enzyme.h ../basecode/Ftype2.h
SparseMatrix.o:	SparseMatrix.h
KineticHub.o:	RateTerm.h Molecule.h Reaction.h Enzyme.h KineticHub.h
Stoich.o:	RateTerm.h Stoich.h SparseMatrix.h Makefile
Kintegrator.o:	Kintegrator.h
TestKinetics.o:	RateTerm.h Molecule.h Reaction.h Enzyme.h KineticHub.h Stoich.h SparseMatrix.h Makefile

#Optional: Comment out the following two items if not using GSL.
$(GSL_OBJ) : $(HEADERS)
GslIntegrator.o:	GslIntegrator.h Stoich.h RateTerm.h SparseMatrix.h 

.cpp.o:
	$(CXX) $(CFLAGS) $(GSL_FLAGS) -I../basecode $< -c

$(TARGET):		$(OBJ) $(GSL_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(GSL_OBJ) $(GSL_LIBS)
#	$(LD) -r -o $(TARGET) $(OBJ) $(GSL_OBJ) $(LIBS) $(GSL_LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
