#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = shell.o

OBJ = \
	Shell.o	\
	ShellCopy.o	\
	ShellThreads.o	\
	Neutral.o	\
	testShell.o	\

HEADERS = \
	../basecode/header.h \


default: $(TARGET)

$(OBJ)	: $(HEADERS)
Shell.o:	Shell.h ../msg/AssignmentMsg.h Neutral.h
ShellCopy.o:	Shell.h Neutral.h
ShellThreads.o:	Shell.h Neutral.h
Neutral.o:	Neutral.h ../basecode/ElementValueFinfo.h
testShell.o:	Shell.h Neutral.h ../scheduling/Tick.h ../scheduling/testScheduling.h ../builtins/Arith.h ../msg/MsgManager.h ../basecode/SparseMatrix.h ../msg/SparseMsg.h ../msg/SingleMsg.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I. -I../basecode -I../msg $< -c

$(TARGET):		$(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
