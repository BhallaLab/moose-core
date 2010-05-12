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

TARGET = basecode.o

#  CXXFLAGS += -I.. -I../kinetics -I../biophysics

OBJ = \
	Element.o	\
	ZeroDimHandler.o \
	OneDimHandler.o \
	ZeroDimGlobalHandler.o \
	OneDimGlobalHandler.o \
	Eref.o	\
	Finfo.o	\
	DestFinfo.o	\
	Cinfo.o	\
	Msg.o	\
	MsgManager.o \
	SingleMsg.o	\
	OneToOneMsg.o	\
	OneToAllMsg.o	\
	SrcFinfo.o \
	SharedFinfo.o \
	Id.o \
	FullId.o \
	Neutral.o	\
	Qinfo.o \
	DataId.o \
	Message.o \
	SetGet.o \
	OpFunc.o \
	SparseMatrix.o \
	PsparseMsg.o \
	DiagonalMsg.o \
	PrepackedBuffer.o \
	AssignmentMsg.o \
	AssignVecMsg.o \
	main.o	\
	testSync.o	\
	testAsync.o	\

HEADERS = \
	header.h \
	Cinfo.h \
	PrepackedBuffer.h \
	Conv.h \
	Data.h \
	DataId.h \
	Dinfo.h \
	DataHandler.h \
	ZeroDimHandler.h \
	OneDimHandler.h \
	ZeroDimGlobalHandler.h \
	OneDimGlobalHandler.h \
	FieldDataHandler.h \
	Element.h \
	Eref.h \
	Finfo.h \
	DestFinfo.h \
	Id.h \
	Msg.h \
	MsgManager.h \
	SingleMsg.h \
	OneToOneMsg.h \
	OneToAllMsg.h \
	Neutral.h \
	OpFunc.h \
	EpFunc.h \
	ProcInfo.h \
	Qinfo.h \
	SrcFinfo.h \
	ValueFinfo.h \
	SharedFinfo.h \
	FieldElementFinfo.h \
	MsgFuncBinding.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS)
testAsync.o:	SparseMatrix.h SetGet.h
SparseMsg.o:	SparseMatrix.h
#TestBasecode.o:	Shell.h ShellWrapper.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
