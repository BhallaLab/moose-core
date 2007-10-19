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
	Id.o	\
	IdManager.o	\
	Conn.o	\
	RecvFunc.o \
	SimpleElement.o	\
	ArrayElement.o	\
	ArrayWrapperElement.o	\
	Copy.o \
	send.o \
	SrcFinfo.o \
	DestFinfo.o \
	ValueFinfo.o \
	LookupFinfo.o \
	ThisFinfo.o \
	DeletionMarkerFinfo.o \
	GlobalMarkerFinfo.o \
	Cinfo.o \
	DynamicFinfo.o \
	UnitTests.o \
	TestBasecode.o \
	strconv.o \
	DerivedFtype.o \
	SharedFtype.o \
	SharedFinfo.o \
	SolveFinfo.o \
	setget.o \
	filecheck.o \
	FunctionData.o \
	ParallelDummy.o \
	Class.o	\

HEADERS = \
	header.h \
	RecvFunc.h \
	Conn.h \
	Ftype.h \
	FunctionData.h \
	Finfo.h \
	Element.h \
	Id.h \
	Class.h \

#	AssignFinfo.h
#	ArrayFinfo.o \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
#TestBasecode.o:	Shell.h ShellWrapper.h

SrcFinfo.o:		SrcFinfo.h
DestFinfo.o: 	DestFinfo.h
ThisFinfo.o: 	ThisFinfo.h
DynamicFinfo.o:	DynamicFinfo.h
ValueFinfo.o:	ValueFinfo.h DynamicFinfo.h send.h
DerivedFtype.o:	DerivedFtype.h
SharedFtype.o:	SharedFtype.h
SharedFinfo.o:	SharedFinfo.h SharedFtype.h
SolveFinfo.o:	ThisFinfo.h SolveFinfo.h
Cinfo.o:		Cinfo.h
UnitTests.o:	MsgSrc.h MsgDest.h SimpleElement.h DerivedFtype.h SrcFinfo.h DestFinfo.h Cinfo.h DynamicFinfo.h ValueFinfo.h send.h ValueFtype.h
LookupFinfo.o:	LookupFinfo.h DynamicFinfo.h LookupFtype.h
DeletionMarkerFinfo.o: DeletionMarkerFinfo.h
GlobalMarkerFinfo.o: DeletionMarkerFinfo.h GlobalMarkerFinfo.h
SimpleElement.o:	SimpleElement.h MsgSrc.h MsgDest.h DeletionMarkerFinfo.h
setget.o:		setget.h DerivedFtype.h
strconv.o:		DerivedFtype.h
filecheck.o: filecheck.h
FunctionData.o:	FunctionData.h
IdManager.o:	Id.h IdManager.h ArrayWrapperElement.h
ArrayElement.o:	ArrayElement.h SimpleElement.h MsgSrc.h MsgDest.h DeletionMarkerFinfo.h
ArrayWrapperElement.o:	ArrayWrapperElement.h SimpleElement.h MsgSrc.h MsgDest.h DeletionMarkerFinfo.h
Class.o: Class.h Cinfo.h Cinfo.cpp
.cpp.o:
	$(CXX) $(CFLAGS) $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
