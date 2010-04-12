#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

OSG_INCLUDE = /usr/include/osg/include
OSG_LIBS = -L/usr/lib -losg -losgUtil -losgDB -losgGA -losgViewer -losgText -lOpenThreads
BOOST_LIBS = -L/lib -lboost_filesystem -lboost_system -lboost_serialization -lboost_thread

MOOSE_HEADERS = \
	../../basecode/header.h \
	../../connections/Conn.h \
	../../basecode/Element.h

DATA_STRUCTURE_HEADERS = AckPickData.h GLcellResetData.h GLcellProcData.h \
	ParticleData.h SmoldynShapeData.h GLviewResetData.h GLshapeData.h

TARGET_CELL = glcell.o
TARGET_VIEW = glview.o
TARGET_SHAPE = glshape.o
TARGET_CLIENT = glclient

COMPARTMENT_DATA_HEADERS = GLCompartmentCylinderData.h GLCompartmentDiskData.h GLCompartmentSphereData.h \
	GLCompartmentHemiData.h GLCompartmentTriData.h GLCompartmentRectData.h

TARGET_CLIENT_SOURCES = GLclient.cpp TextBox.cpp GLCompartment.cpp \
	GLCompartmentCylinder.cpp GLCompartmentSphere.cpp GLCompartmentHemi.cpp \
	GLCompartmentDisk.cpp GLCompartmentTri.cpp GLCompartmentRect.cpp GLviewShape.cpp

TARGET_CLIENT_HEADERS = GLclient.h TextBox.h GLCompartment.h GLCompartmentType.h \
	GLCompartmentCylinder.h GLCompartmentSphere.h GLCompartmentHemi.h \
	GLCompartmentDisk.h GLCompartmentTri.h GLCompartmentRect.h \
	$(COMPARTMENT_DATA_HEADERS) ParticleData.h SmoldynShapeData.h GLviewShape.h

default: $(TARGET_CELL) $(TARGET_VIEW) $(TARGET_SHAPE) $(TARGET_CLIENT) 

clean:
	rm -f *.o $(TARGET_CELL) $(TARGET_VIEW) $(TARGET_SHAPE) $(TARGET_CLIENT) core core.*

OBJ_CELL = GLcell.o
$(OBJ_CELL) : GLcell.cpp GLcell.h $(DATA_STRUCTURE_HEADERS) $(MOOSE_HEADERS) $(COMPARTMENT_DATA_HEADERS)
	$(CXX) $(CXXFLAGS) -I../../ -I../../basecode -I$(OSG_INCLUDE) -c GLcell.cpp
$(TARGET_CELL) : $(OBJ_CELL) $(MOOSE_HEADERS)
	$(LD) -r -o $(TARGET_CELL) $(OBJ_CELL)

OBJ_SHAPE = GLshape.o
$(OBJ_SHAPE) : GLshape.cpp GLshape.h $(MOOSE_HEADERS)
	$(CXX) $(CXXFLAGS) -I../../ -I../../basecode -I$(OSG_INCLUDE) -c GLshape.cpp
$(TARGET_SHAPE) : $(OBJ_SHAPE) $(MOOSE_HEADERS)
	$(LD) -r -o $(TARGET_SHAPE) $(OBJ_SHAPE)

OBJ_VIEW = GLview.o
$(OBJ_VIEW) : GLview.cpp GLview.h $(DATA_STRUCTURE_HEADERS) $(MOOSE_HEADERS)
	$(CXX) $(CXXFLAGS) -I../../ -I../../basecode -I$(OSG_INCLUDE) -c GLview.cpp
$(TARGET_VIEW) : $(OBJ_VIEW) $(MOOSE_HEADERS)
	$(LD) -r -o $(TARGET_VIEW) $(OBJ_VIEW)

$(TARGET_CLIENT) : $(TARGET_CLIENT_SOURCES) $(TARGET_CLIENT_HEADERS) $(DATA_STRUCTURE_HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET_CLIENT) $(TARGET_CLIENT_SOURCES) $(BOOST_LIBS) $(OSG_LIBS)
