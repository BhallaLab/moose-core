#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = kinetics.o

OBJ = \
	Molecule.o	\
	Reaction.o	\
	Enzyme.o	\
	MathFunc.o	\
	Geometry.o	\
	Surface.o	\
	Panel.o	\
	RectPanel.o	\
	TriPanel.o	\
	SpherePanel.o	\
	HemispherePanel.o	\
	CylPanel.o	\
	DiskPanel.o	\
	KinCompt.o	\
	KinPlaceHolder.o	\
	TestKinetics.o	\

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \


# Smoldyn, written by Steven Andrews, provides a numerical engine for
# 3-D single-particle simulations using Smoluchowsky dynamics. It is
# licenced on a non-commercial license. Should be fine for academic use.
# Optional: Comment out the following three items if not using Smoldyn
ifdef ($(USE_SMOLDYN))
SMOLDYN_OBJ = \
	Particle.o \
	SmoldynHub.o \
SMOLDYN_FLAGS = -DUSE_SMOLDYN
# This is a static lib version, does not need the lib_path
#SMOLDYN_LIBS = /home/bhalla/genesis/SFmus/moose-g3/kinetics/Smoldyn/source/libsmoldyn.a

# This version uses dynamic libs. Not yet confirmed to work.
#SMOLDYN_LIB_PATH = -L/home/bhalla/genesis/SFmus/moose-g3/kinetics/Smoldyn/source
#SMOLDYN_LIBS = libsmoldyn.so
endif # !USE_SMOLDYN

default: $(TARGET)

$(OBJ)	: $(HEADERS)
Molecule.o:	Molecule.h ../basecode/Ftype2.h
Reaction.o:	Reaction.h ../basecode/Ftype2.h
Enzyme.o:	Enzyme.h ../basecode/Ftype2.h
Geometry.o:	Geometry.h
Surface.o:	Surface.h
Panel.o:	Panel.h
RectPanel.o:	Panel.h RectPanel.h
TriPanel.o:	Panel.h TriPanel.h
SpherePanel.o:	Panel.h SpherePanel.h
HemispherePanel.o:	Panel.h HemispherePanel.h
CylPanel.o:	Panel.h CylPanel.h
DiskPanel.o:	Panel.h DiskPanel.h
KinCompt.o:	KinCompt.h
KinPlaceHolder.o:	KinPlaceHolder.h
TestKinetics.o:	Molecule.h Reaction.h Enzyme.h Makefile

ifdef ($(USE_SMOLDYN)
$(SMOLDYN_OBJ) : $(HEADERS)
Particle.o:		SmoldynHub.h Particle.h Molecule.h
SmoldynHub.o:	KinSparseMatrix.h SmoldynHub.h Smoldyn/source/smolrun.h
endif # !USE_SMOLDYN
.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I.. -I../basecode -I../element $< -c

$(TARGET):		$(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) 

clean:
	-rm -f *.o $(TARGET) core core.*
