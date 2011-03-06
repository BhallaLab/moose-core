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
	ShellSetGet.o	\
	ShellThreads.o	\
	ProcessLoop.o	\
	LoadModels.o \
	Neutral.o	\
	Wildcard.o	\
	testShell.o	\

HEADERS = \
	../basecode/header.h \
	../basecode/ReduceBase.h \
	../basecode/ReduceMax.h \


default: $(TARGET)

$(OBJ)	: $(HEADERS)
Shell.o:	Shell.h ../msg/AssignmentMsg.h Neutral.h ../scheduling/Clock.h 
ShellCopy.o:	Shell.h Neutral.h
ShellSetGet.o:	Shell.h
ShellThreads.o:	Shell.h Neutral.h ../scheduling/Clock.h
ProcessLoop.o:	Shell.h FuncBarrier.h
LoadModels.o:	Shell.h Neutral.h
Neutral.o:	Neutral.h ../basecode/ElementValueFinfo.h
Wildcard.o:	Wildcard.h Shell.h Neutral.h ../basecode/ElementValueFinfo.h
testShell.o:	Wildcard.h Shell.h Neutral.h ../scheduling/Tick.h ../scheduling/testScheduling.h ../builtins/Arith.h ../basecode/SparseMatrix.h ../msg/SparseMsg.h ../msg/SingleMsg.h ../basecode/SetGet.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I. -I../basecode -I../msg $< -c

$(TARGET):		$(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
