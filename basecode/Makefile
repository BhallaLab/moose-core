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
	Conn.o	\
	Cinfo.o	\
	CinfoWrapper.o	\
	Ftype.o \
	Field.o	\
	Finfo.o	\
	RelayFinfo.o	\
	ValueFinfo.o	\
	DestFinfo.o	\
	SingleSrcFinfo.o	\
	NSrcFinfo.o	\
	SharedFinfo.o \
	ReturnFinfo.o \
	ObjFinfo.o \
	MsgSrc.o	\
	MsgFuncs.o	\
	Element.o	\
	Neutral.o	\
	Shell.o	\
	ShellWrapper.o	\
	ReadCell.o	\
	Wildcard.o	\
	TestBasecode.o \

#	AssignFinfo.o


HEADERS = \
	header.h \
	Conn.h \
	Cinfo.h \
	Ftype.h \
	Field.h \
	Finfo.h \
	MsgSrc.h \
	RelayFinfo.h \
	ValueFinfo.h \
	DestFinfo.h \
	SingleSrcFinfo.h \
	NSrcFinfo.h \
	ReturnFinfo.h \
	SharedFinfo.h \
	ObjFinfo.h \
	Element.h \
	Neutral.h \
	ProcInfo.h \

#	AssignFinfo.h

default: $(TARGET)

$(OBJ)	: $(HEADERS)
Cinfo.o: CinfoWrapper.h
CinfoWrapper.o: CinfoWrapper.h
Shell.o:  Shell.h
ShellWrapper.o:	Shell.h ShellWrapper.h
ReadCell.o:	Shell.h ShellWrapper.h ../builtins/Interpol.h
TestBasecode.o:	Shell.h ShellWrapper.h

.cpp.o:
	$(CXX) $(CFLAGS) $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
