#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = msg.o

OBJ = \
	Msg.o	\
	AssignmentMsg.o	\
	AssignVecMsg.o	\
	DiagonalMsg.o	\
	MsgManager.o	\
	OneToAllMsg.o	\
	OneToOneMsg.o	\
	SingleMsg.o	\
	SparseMsg.o	\
	testMsg.o	\

HEADERS = \
	../basecode/header.h \


default: $(TARGET)

$(OBJ)	: $(HEADERS)
AssignmentMsg.o:	AssignmentMsg.h
AssignVecMsg.o:	AssignVecMsg.h
DiagonalMsg.o:	DiagonalMsg.h
MsgManager.o:	MsgManager.h
OneToAll.o:	OneToAll.h
OneToOne.o:	OneToOne.h
SingleMsg.o:	SingleMsg.h
SparseMsg.o:	SparseMsg.h
testMsg.o: AssignmentMsg.h AssignVecMsg.h DiagonalMsg.h MsgManager.h OneToAllMsg.h OneToOneMsg.h SingleMsg.h SparseMsg.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I. -I../basecode $< -c

$(TARGET):		$(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
