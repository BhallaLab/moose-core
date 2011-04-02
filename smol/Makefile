#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = smol.o

OBJ = \
	SmolSim.o	\
	SmolPool.o \
	SmolMol.o \
	SmolReac.o \
	SmolEnz.o \
	SmolMMenz.o \
	testSmol.o	\

	
	#KineticHub.o	\
	#Tab.o	\

HEADERS = \
	../basecode/header.h \
	../basecode/ElementValueFinfo.h \
	SmolSim.h \
	../kinetics/Mol.h \


# Smoldyn is GPL so only GPL-compliant uses of MOOSE will be able to
# use it. Should be fine for academic use.

default: $(TARGET)

$(OBJ)	: $(HEADERS)
SmolSim.o:	SmolMol.h SmolReac.h SmolEnz.h SmolMMenz.h ../kinetics/Mol.h ../kinetics/Reac.h ../kinetics/Enz.h ../kinetics/MMenz.h ../kinetics/Boundary.h ../kinetics/ChemCompt.h
SmolPool.o:	SmolPool.h
SmolMol.o:	SmolMol.h
SmolReac.o:	../kinetics/Reac.h SmolReac.h
SmolEnz.o:	SmolEnz.h
SmolMMenz.o: SmolMMenz.h
testSmol.o:	SmolPool.h SmolMol.h SmolReac.h SmolEnz.h SmolMMenz.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(GSL_FLAGS) $(SMOLDYN_FLAGS) -I.. -I../basecode -I../msg -I../kinetics $< -c

$(TARGET):		$(OBJ) $(GSL_OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(GSL_OBJ) $(GSL_LIBS) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
