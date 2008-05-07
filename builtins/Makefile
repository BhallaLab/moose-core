#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = builtins.o

OBJ = \
	Interpol.o	\
	Table.o	\


ifneq (,$(findstring -DCRL_MPI,$(CFLAGS)))
	OBJ += ParTable.o
endif


HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
Interpol.o:	Interpol.h ../basecode/Ftype2.h
Table.o:	Table.h Interpol.h ../basecode/Ftype2.h ../basecode/Ftype3.h

ifneq (,$(findstring -DCRL_MPI,$(CFLAGS)))
ParTable.o:	ParTable.h Table.h Interpol.h ../basecode/Ftype2.h
endif

.cpp.o:
	$(CXX) $(CFLAGS) -I../basecode $< -c

#$(TARGET):		$(OBJ) $(HEADERS)
#	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
