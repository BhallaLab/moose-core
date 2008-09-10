#$/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************$/

LOCAL_DIR := hsolve

LOCAL_SRCS := \
	HSolveStruct.cpp	\
 	HinesMatrix.cpp \
 	HSolvePassive.cpp	\
	RateLookup.cpp	\
 	HSolveActive.cpp	\
	HSolveInterface.cpp \
 	HSolve.cpp \
 	HSolveHub.cpp \
	TestHSolve.cpp

#
# Dependencies
#

$(LOCAL_DIR)$/HSolveStruct.o: \
	$(LOCAL_DIR)$/HSolveStruct.h \
	$(LOCAL_DIR)$/RateLookup.h 

$(LOCAL_DIR)$/HinesMatrix.o: \
	$(LOCAL_DIR)$/HinesMatrix.h \
	$(LOCAL_DIR)$/TestHSolve.h \
	utility$/utility.h

$(LOCAL_DIR)$/HSolvePassive.o: \
	$(LOCAL_DIR)$/HSolvePassive.h \
	$(LOCAL_DIR)$/HinesMatrix.h \
	$(LOCAL_DIR)$/HSolveStruct.h \
	biophysics$/BioScan.h \
	$(LOCAL_DIR)$/TestHSolve.h \
	utility$/utility.h

$(LOCAL_DIR)$/RateLookup.o: \
	$(LOCAL_DIR)$/RateLookup.h

$(LOCAL_DIR)$/HSolveActive.o: \
	$(LOCAL_DIR)$/HSolveActive.h \
	$(LOCAL_DIR)$/RateLookup.h \
	$(LOCAL_DIR)$/HSolvePassive.h \
	$(LOCAL_DIR)$/HinesMatrix.h \
	$(LOCAL_DIR)$/HSolveStruct.h \
	biophysics$/BioScan.h \
	biophysics$/SynChan.h \
	biophysics$/SynInfo.h \
	biophysics$/SpikeGen.h

$(LOCAL_DIR)$/HSolveInterface.o: \
	$(LOCAL_DIR)$/HSolveActive.h \
	$(LOCAL_DIR)$/RateLookup.h \
	$(LOCAL_DIR)$/HSolvePassive.h \
	$(LOCAL_DIR)$/HinesMatrix.h \
	$(LOCAL_DIR)$/HSolveStruct.h

$(LOCAL_DIR)$/HSolve.o:	\
	$(LOCAL_DIR)$/HSolve.h \
	$(LOCAL_DIR)$/HSolveActive.h \
	$(LOCAL_DIR)$/RateLookup.h \
	$(LOCAL_DIR)$/HSolvePassive.h \
	$(LOCAL_DIR)$/HinesMatrix.h \
	$(LOCAL_DIR)$/HSolveStruct.h \
	element$/Neutral.h

$(LOCAL_DIR)$/HSolveHub.o: \
	$(LOCAL_DIR)$/HSolveHub.h \
	basecode$/SolveFinfo.h \
	basecode$/ThisFinfo.h \
	$(LOCAL_DIR)$/HSolveActive.h \
	$(LOCAL_DIR)$/RateLookup.h \
	$(LOCAL_DIR)$/HSolvePassive.h \
	$(LOCAL_DIR)$/HinesMatrix.h \
	$(LOCAL_DIR)$/HSolveStruct.h \
	biophysics$/Compartment.h \
	biophysics$/HHChannel.h \
	biophysics$/CaConc.h \
	element$/Neutral.h

LOCAL_HEADERS := 	\
	HSolveStruct.h	\
 	HinesMatrix.h \
 	HSolvePassive.h	\
	RateLookup.h	\
 	HSolveActive.h	\
 	HSolve.h \
 	HSolveHub.h \
	TestHSolve.h

# End of dependencies

SOURCES += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_SRCS))

HEADERS += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_HEADERS))

