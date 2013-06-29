#/**********************************************************************
# ** This program is part of 'MOOSE', the
# ** Multiscale Object Oriented Simulation Environment.
# **   copyright (C) 2003-2011 Upinder S. Bhalla, Niraj Dudani and NCBS
# ** It is made available under the terms of the
# ** GNU Lesser General Public License version 2.1
# ** See the file COPYING.LIB for the full notice.
# **********************************************************************/
 
OBJS =	\
 	launchHines.o \

TARGET = testHines

$(TARGET):	$(OBJS)
		g++ -Wall -Wl,-rpath,../cudaLibrary -o $(TARGET) $(OBJS) ../cudaLibrary/libmooseCudaLibrary.so -lpthread 

default: $(TARGET)

%.o: %.cpp
	g++ $(CXXFLAGS) -c -o "$@" "$<"
	
clean:
	rm -f $(OBJS) $(TARGET)
