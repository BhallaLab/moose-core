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
	ParShell.o	\
	ReadCell.o \
	SimDump.o \
	LoadTab.o \
	TestShell.o	\

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)

Shell.o TestShell.o: Shell.h ../randnum/Probability.h ../randnum/Uniform.h ../randnum/Exponential.h ../randnum/Normal.h

# ifneq (,$(findstring -DCRL_MPI,$(CXXFLAGS)))
#endif

ParShell.o: Shell.h ParShell.h
ReadCell.o: ReadCell.h Shell.h


.cpp.o:
	$(CXX) $(CXXFLAGS) -I../external/include -I/usr/include/libxml2 -I.. -I../basecode $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)


clean:
	-rm -f *.o $(TARGET) core core.* 

