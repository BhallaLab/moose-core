#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

TARGET = _hsolve.o

OBJ = \
	Cell.o \
	HSolveStruct.o \
	HinesMatrix.o \
	HSolvePassive.o \
	RateLookup.o \
	HSolveActive.o \
	HSolveActiveSetup.o \
	HSolveInterface.o \
	HSolve.o \
	HSolveUtils.o \
	testHSolve.o \
	ZombieCompartment.o \

	#~ HSolveHub.o \

HEADERS = \
	../basecode/header.h

default: $(TARGET)

#~ $(OBJ)	: $(HEADERS)
#~ HSolveStruct.o:	HSolveStruct.h
#~ HinesMatrix.o:	HinesMatrix.h TestHSolve.h ../utility/utility.h
#~ HSolvePassive.o:	HSolvePassive.h HinesMatrix.h HSolveStruct.h ../biophysics/BioScan.h TestHSolve.h ../utility/utility.h
#~ RateLookup.o:	RateLookup.h
#~ HSolveActive.o:	HSolveActive.h RateLookup.h HSolvePassive.h HinesMatrix.h HSolveStruct.h ../biophysics/SynChan.h ../biophysics/SynInfo.h ../biophysics/SpikeGen.h
#~ HSolveActiveSetup.o:	HSolveActive.h RateLookup.h HSolvePassive.h HinesMatrix.h HSolveStruct.h ../biophysics/BioScan.h
#~ HSolveInterface.o:	HSolveActive.h RateLookup.h HSolvePassive.h HinesMatrix.h HSolveStruct.h
#~ HSolve.o:	HSolve.h HSolveActive.h RateLookup.h HSolvePassive.h HinesMatrix.h HSolveStruct.h ../element/Neutral.h
#~ HSolveHub.o:	HSolveHub.h ../basecode/SolveFinfo.h ../basecode/ThisFinfo.h HSolveActive.h RateLookup.h HSolvePassive.h HinesMatrix.h HSolveStruct.h ../biophysics/Compartment.h ../biophysics/HHChannel.h ../biophysics/CaConc.h ../element/Neutral.h

$(OBJ)	: $(HEADERS)
Cell.o:	Cell.h ../shell/Shell.h
HSolve.o:	HSolve.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $(SMOLDYN_FLAGS) -I. -I../basecode -I../msg $< -c

$(TARGET): $(OBJ) $(SMOLDYN_OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(SMOLDYN_OBJ) $(SMOLDYN_LIB_PATH) $(SMOLDYN_LIBS) $(GSL_LIBS)

clean:
	rm -f *.o $(TARGET) core core.*
