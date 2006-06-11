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

TARGET = builtins.o

OBJ = \
	String.o	\
	Int.o	\
	Double.o	\
	Interpol.o	\
	InterpolWrapper.o	\
	TableWrapper.o	\


HEADERS = \
	../basecode/header.h \
	../basecode/Conn.h \
	../basecode/Cinfo.h \
	../basecode/Ftype.h \
	../basecode/Field.h \
	../basecode/Finfo.h \
	../basecode/MsgSrc.h \
	../basecode/RelayFinfo.h \
	../basecode/ValueFinfo.h \
	../basecode/DestFinfo.h \
	../basecode/SingleSrcFinfo.h \
	../basecode/NSrcFinfo.h \
	../basecode/ReturnFinfo.h \
	../basecode/SharedFinfo.h \
	../basecode/ObjFinfo.h \
	../basecode/Element.h \
	../basecode/Neutral.h \
	../basecode/ProcInfo.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
String.o: String.h
Int.o: Int.h
Double.o: Double.h
Interpol.o: Interpol.h
InterpolWrapper.o: Interpol.h InterpolWrapper.h
TableWrapper.o: Interpol.h Table.h TableWrapper.h

.cpp.o:
	$(CXX) $(CFLAGS) -I../basecode $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
