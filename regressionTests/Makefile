#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = rt.o

OBJ = \
	regressionTest.o	\
	rtTable.o	\
	rtReadKkit.o	\
	rtHHnetwork.o	\
	benchmarkTests.o	\

HEADERS = \
	../basecode/header.h \
	../shell/Shell.h

default: $(TARGET)

$(OBJ)	: $(HEADERS)
rtReadKkit.o:	../kinetics/ReadKkit.h ../shell/LoadModels.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I. -I../basecode -I../msg -I../shell $< -c

$(TARGET):		$(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
