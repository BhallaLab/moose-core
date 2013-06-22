#/**********************************************************************
# ** This program is part of 'MOOSE', the
# ** Multiscale Object Oriented Simulation Environment.
# **   copyright (C) 2003-2011 Upinder S. Bhalla, Niraj Dudani and NCBS
# ** It is made available under the terms of the
# ** GNU Lesser General Public License version 2.1
# ** See the file COPYING.LIB for the full notice.
# **********************************************************************/
 
CXXFLAGS =	-g -lm -I/usr/include/python2.7

OBJS =	\
 	testMooseCuda.o \

TARGET = testMooseCuda

$(TARGET):	$(OBJS)
		g++ -o $(TARGET) $(OBJS) $(LIBS) -lm -lpython2.7 -L/usr/lib/python2.7/config -L../cudaLibrary 

default: $(TARGET)

%.o: %.cpp
	g++ $(CXXFLAGS) -c -o "$@" "$<"
clean:
	rm -f $(OBJS) $(TARGET)
