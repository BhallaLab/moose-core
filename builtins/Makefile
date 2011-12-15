#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = _builtins.o

OBJ = \
	Arith.o	\
	Group.o	\
	Mdouble.o	\
	Table.o	\
	TableEntry.o	\
	Stats.o	\
	Interpol2D.o \
	testBuiltins.o	\

HEADERS = \
	../basecode/header.h \


default: $(TARGET)

$(OBJ)	: $(HEADERS)
Arith.o:	Arith.h
Group.o:	Group.h
Mdouble.o:		Mdouble.h
Table.o:		Table.h TableEntry.h
TableEntry.o:		TableEntry.h
Stats.o:	Stats.h
Interpol2D.o:	Interpol2D.h
testBuiltins.o:	Group.h Arith.h Stats.h ../msg/DiagonalMsg.h ../basecode/SetGet.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I. -I../basecode -I../msg $< -c

$(TARGET):		$(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
