#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/
# $Id: $
#
# $Log: $
#

TARGET = connections.o

#CXXFLAGS += -I.. -I../basecode

OBJ = \
	ConnTainer.o	\
	SimpleConn.o	\
	One2AllConn.o	\
	All2OneConn.o	\
	Many2ManyConn.o	\
	One2OneMapConn.o	\
	TraverseMsgConn.o	\
	TraverseDestConn.o	\

HEADERS = \
	../basecode/header.h \
	../basecode/Eref.h \
	../basecode/RecvFunc.h \
	Conn.h \
	ConnTainer.h \
	../basecode/Ftype.h \
	../basecode/FuncVec.h \
	../basecode/Slot.h \
	../basecode/Finfo.h \
	../basecode/Id.h \
	../basecode/Msg.h \
	../basecode/Element.h \
	../basecode/Class.h \

INCLUDE = -I../basecode -I../element

default: $(TARGET)

$(OBJ)	: $(HEADERS) ConnTainer.h
#TestBasecode.o:	Shell.h ShellWrapper.h

ConnTainer.o: ConnTainer.h
SimpleConn.o: SimpleConn.h
One2AllConn.o: One2AllConn.h
All2OneConn.o: All2OneConn.h
Many2ManyConn.o: Many2ManyConn.h ../utility/SparseMatrix.h
TraverseMsgConn.o: TraverseMsgConn.h
TraverseDestConn.o: TraverseDestConn.h
One2OneMapConn.o: One2OneMapConn.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE) $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
