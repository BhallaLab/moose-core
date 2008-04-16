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

TARGET = basecode.o
CFLAGS += -I..
OBJ = \
	Element.o	\
	Eref.o	\
	Id.o	\
	IdManager.o	\
	RecvFunc.o \
	Msg.o \
	SimpleConn.o	\
	TraverseMsgConn.o	\
	TraverseDestConn.o	\
	SimpleElement.o	\
	ArrayElement.o \
	Copy.o \
	Send.o \
	SrcFinfo.o \
	DestFinfo.o \
	ValueFinfo.o \
	LookupFinfo.o \
	ThisFinfo.o \
	DeletionMarkerFinfo.o \
	GlobalMarkerFinfo.o \
	Cinfo.o \
	DynamicFinfo.o \
	strconv.o \
	DerivedFtype.o \
	SharedFtype.o \
	SharedFinfo.o \
	SolveFinfo.o \
	setget.o \
	filecheck.o \
	ParallelDummy.o \
	Class.o	\
	FuncVec.o \
	ExtFieldFinfo.o \
	UnitTests.o \
	TestBasecode.o \

HEADERS = \
	header.h \
	Eref.h \
	RecvFunc.h \
	Conn.h \
	ConnTainer.h \
	Ftype.h \
	FuncVec.h \
	Slot.h \
	Finfo.h \
	Id.h \
	Msg.h \
	Element.h \
	Class.h \
#	AssignFinfo.h
#	ArrayFinfo.o \

#	ArrayWrapperElement.o	\

default: $(TARGET)

$(OBJ)	: $(HEADERS)
#TestBasecode.o:	Shell.h ShellWrapper.h

SrcFinfo.o:		SrcFinfo.h
DestFinfo.o: 	DestFinfo.h
ThisFinfo.o: 	ThisFinfo.h
DynamicFinfo.o:	DynamicFinfo.h
ValueFinfo.o:	ValueFinfo.h DynamicFinfo.h Send.h
ExtFieldFinfo.o:	ExtFieldFinfo.h
DerivedFtype.o:	DerivedFtype.h
SharedFtype.o:	SharedFtype.h
SharedFinfo.o:	SharedFinfo.h SharedFtype.h
SolveFinfo.o:	ThisFinfo.h SolveFinfo.h
Cinfo.o:		Cinfo.h
UnitTests.o:	SimpleElement.h DerivedFtype.h SrcFinfo.h DestFinfo.h Cinfo.h DynamicFinfo.h ValueFinfo.h Send.h ValueFtype.h
LookupFinfo.o:	LookupFinfo.h DynamicFinfo.h LookupFtype.h
DeletionMarkerFinfo.o: DeletionMarkerFinfo.h
GlobalMarkerFinfo.o: DeletionMarkerFinfo.h GlobalMarkerFinfo.h
SimpleElement.o:	SimpleElement.h DeletionMarkerFinfo.h
setget.o:		setget.h DerivedFtype.h
strconv.o:		DerivedFtype.h
filecheck.o: filecheck.h
IdManager.o:	Id.h IdManager.h ArrayWrapperElement.h
ArrayElement.o:	ArrayElement.h SimpleElement.h DeletionMarkerFinfo.h
ArrayWrapperElement.o:	ArrayWrapperElement.h SimpleElement.h DeletionMarkerFinfo.h GlobalMarkerFinfo.h ThisFinfo.h
Class.o: Class.h Cinfo.h Cinfo.cpp
FuncVec.o: RecvFunc.h Ftype.h FuncVec.h
TraverseMsgConn.o: TraverseMsgConn.h
TraverseDestConn.o: TraverseDestConn.h

.cpp.o:
	$(CXX) $(CFLAGS) $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
