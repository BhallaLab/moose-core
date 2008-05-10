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
	KinSparseMatrix.o	\
	KineticHub.o	\
	Stoich.o	\
	Kintegrator.o	\
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
	TestKinetics.o	\
	KineticManager.o	\

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \


# The GSL library (GNU Scientific Library) provides a range of
# numerical functions, which allow us to use various advanced integration
# methods. The GslIntegrator accesses these. This library is licenced
# under the GPL, so only GPL-compliant uses of MOOSE will be able to
# use it. Should be fine for academic use.
# Optional: Comment out the following three items if not using GSL
GSL_OBJ = GslIntegrator.o
GSL_LIBS = -lgsl -lgslcblas
GSL_FLAGS = -DUSE_GSL

# Smoldyn, written by Steven Andrews, provides a numerical engine for
# 3-D single-particle simulations using Smoluchowsky dynamics. It is
# licenced on a non-commercial license. Should be fine for academic use.
# Optional: Comment out the following three items if not using Smoldyn
#SMOLDYN_OBJ = \
	Particle.o \
	SmoldynHub.o \


#SMOLDYN_FLAGS = -DUSE_SMOLDYN
# This is a static lib version, does not need the lib_path
#SMOLDYN_LIBS = /home/bhalla/genesis/SFmus/moose-g3/kinetics/Smoldyn/source/libsmoldyn.a

# This version uses dynamic libs. Not yet confirmed to work.
#SMOLDYN_LIB_PATH = -L/home/bhalla/genesis/SFmus/moose-g3/kinetics/Smoldyn/source
#SMOLDYN_LIBS = libsmoldyn.so

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
KinSparseMatrix.o:	KinSparseMatrix.h
KineticHub.o:	RateTerm.h Molecule.h Reaction.h Enzyme.h KineticHub.h
Stoich.o:	RateTerm.h Stoich.h KinSparseMatrix.h Makefile
Kintegrator.o:	Kintegrator.h
KineticManager.o:	KineticManager.h
TestKinetics.o:	RateTerm.h Molecule.h Reaction.h Enzyme.h KineticHub.h Stoich.h KinSparseMatrix.h Makefile

#Optional: Comment out the following two items if not using GSL.
$(GSL_OBJ) : $(HEADERS)
GslIntegrator.o:	GslIntegrator.h Stoich.h RateTerm.h KinSparseMatrix.h 

$(SMOLDYN_OBJ) : $(HEADERS)
Particle.o:		SmoldynHub.h Particle.h Molecule.h
SmoldynHub.o:	KinSparseMatrix.h SmoldynHub.h Smoldyn/source/smolrun.h

.cpp.o:
	$(CXX) $(CFLAGS) $(GSL_FLAGS) $(SMOLDYN_FLAGS) -I../basecode $< -c

$(TARGET):		$(OBJ) $(GSL_OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(GSL_OBJ) $(SMOLDYN_OBJ) $(GSL_LIBS) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
