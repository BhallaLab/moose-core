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

TARGET = scheduling.o

OBJ = \
	SchedWrapper.o	\
	JobWrapper.o	\
	ClockJobWrapper.o	\
	ClockTickWrapper.o	\
	TockWrapper.o	\
	TestScheduling.o \


HEADERS = \
	../basecode/header.h \
	../basecode/Conn.h \
	../basecode/Cinfo.h \
	../basecode/Ftype.h \
	../basecode/Field.h \
	../basecode/Finfo.h \
	../basecode/MsgSrc.h \
	../basecode/RelayFinfo.h \
	../basecode/ValueFinfo.h \
	../basecode/DestFinfo.h \
	../basecode/SingleSrcFinfo.h \
	../basecode/NSrcFinfo.h \
	../basecode/ReturnFinfo.h \
	../basecode/SharedFinfo.h \
	../basecode/ObjFinfo.h \
	../basecode/Element.h \
	../basecode/Neutral.h \
	../basecode/ProcInfo.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
SchedWrapper.o: Sched.h SchedWrapper.h
JobWrapper.o: Job.h JobWrapper.h
ClockJobWrapper.o: Job.h ClockJob.h ClockJobWrapper.h
ClockTickWrapper.o: ClockTick.h ClockTickWrapper.h
TockWrapper.o: Tock.h TockWrapper.h

.cpp.o:
	$(CXX) $(CFLAGS) -I../basecode $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
