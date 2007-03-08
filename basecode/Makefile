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

OBJ = \
	Element.o	\
	Conn.o	\
	RecvFunc.o \
	SimpleElement.o	\
	send.o \
	SrcFinfo.o \
	DestFinfo.o \
	ValueFinfo.o \
	ArrayFinfo.o \
	LookupFinfo.o \
	ThisFinfo.o \
	DeletionMarkerFinfo.o \
	Cinfo.o \
	DynamicFinfo.o \
	UnitTests.o \
	TestBasecode.o \
	SharedFtype.o \
	SharedFinfo.o \
	setget.o \
	strconv.o \

HEADERS = \
	header.h \
	Conn.h \
	Element.h \
	Ftype.h \

#	AssignFinfo.h

default: $(TARGET)

$(OBJ)	: $(HEADERS)
#TestBasecode.o:	Shell.h ShellWrapper.h

SrcFinfo.o:		SrcFinfo.h
DestFinfo.o: 	DestFinfo.h
ThisFinfo.o: 	ThisFinfo.h
DynamicFinfo.o:	DynamicFinfo.h
ValueFinfo.o:	ValueFinfo.h DynamicFinfo.h send.h
SharedFtype.o:	SharedFtype.h
SharedFinfo.o:	SharedFinfo.h SharedFtype.h
Cinfo.o:		Cinfo.h
UnitTests.o:	MsgSrc.h MsgDest.h SimpleElement.h DerivedFtype.h SrcFinfo.h DestFinfo.h Cinfo.h DynamicFinfo.h ValueFinfo.h send.h ArrayFtype.h ValueFtype.h NestFtype.h
ArrayFinfo.o:	ArrayFinfo.h DynamicFinfo.h ArrayFtype.h
LookupFinfo.o:	LookupFinfo.h DynamicFinfo.h LookupFtype.h
DeletionMarkerFinfo.o: DeletionMarkerFinfo.h
SimpleElement.o:	SimpleElement.h MsgSrc.h MsgDest.h DeletionMarkerFinfo.h
setget.o:		setget.h DerivedFtype.h
strconv.o:		DerivedFtype.h

.cpp.o:
	$(CXX) $(CFLAGS) $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
