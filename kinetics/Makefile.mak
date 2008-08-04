#$/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************$/

LOCAL_DIR := kinetics

LOCAL_SRCS := \
	Molecule.cpp	\
	Reaction.cpp	\
	Enzyme.cpp	\
	KinSparseMatrix.cpp	\
	KineticHub.cpp	\
	Stoich.cpp	\
	Kintegrator.cpp	\
	MathFunc.cpp	\
	Geometry.cpp	\
	Surface.cpp	\
	Panel.cpp	\
	RectPanel.cpp	\
	TriPanel.cpp	\
	SpherePanel.cpp	\
	HemispherePanel.cpp	\
	CylPanel.cpp	\
	DiskPanel.cpp	\
	KinCompt.cpp	\
	TestKinetics.cpp	\
	KineticManager.cpp	\

LOCAL_HEADERS := $(subst .cpp,.h, $(LOCAL_SOURCES))

SOURCES += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_SRCS))

HEADERS += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_HEADERS))
