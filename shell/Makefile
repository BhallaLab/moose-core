#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/
#
# Here we do extra dependencies to get SWIG to compile. The
# final stage of compilation and assorted flags are to be set
# in the moose root Makefile.
#

TARGET = shell.o

OBJ = \
	Shell.o	\
	SwigFuncs.o	\

HEADERS = \
	../basecode/header.h \
	../basecode/Conn.h \
	../basecode/Element.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)

Swig_wrap.o: Swig.h Shell.h Swig_wrap.cxx
Shell.o: Shell.h
SwigFuncs.o: Swig.h Shell.h

Swig_wrap.cxx:
	swig -c++ -python Swig.i

Swig_wrap.o:
	g++ -fpic -c Swig_wrap.cxx -I/usr/include/python2.4

#MyClassWrapper.o: MyClassWrapper.h

.cpp.o:
	$(CXX) $(CFLAGS) -I../basecode $< -c

$(TARGET):		$(OBJ) $(HEADERS) Swig_wrap.o
	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)
	cp pymoose.py ..

clean:
	-rm -f *.o $(TARGET) core core.* Swig_wrap.cxx pymoose.py

