#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = _geom.o

OBJ = \
	Geometry.o	\
	Surface.o	\
	Panel.o \
	CylPanel.o	\
	SpherePanel.o	\
	HemispherePanel.o	\
	DiskPanel.o	\
	RectPanel.o	\
	TriPanel.o	\
	testGeom.o	\

HEADERS = \
	../basecode/header.h \
	Panel.h


default: $(TARGET)

$(OBJ)	: $(HEADERS)
Geometry.o:	Geometry.h
Surface.o:	Surface.h
CylPanel.o:	CylPanel.h
SpherePanel.o:	SpherePanel.h
HemispherePanel.o:	HemispherePanel.h
DiskPanel.o:	DiskPanel.h
RectPanel.o:	RectPanel.h 
TriPanel.o:	TriPanel.h
testGeom.o:	Geometry.h

.cpp.o:
	$(CXX) $(CXXFLAGS) -I.. -I../basecode -I../msg $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
