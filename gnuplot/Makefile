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

TARGET = gnuplot_i.o

HEADERS = \
	gnuplot_i.h \
	
        

default: $(TARGET)

$(OBJ)	: $(HEADERS)


.cpp.o:
	$(CXX) $(CFLAGS)  $< -c

$(TARGET):		$(OBJ) $(HEADERS)

clean:
	-rm -f *.o $(TARGET) core core.*
