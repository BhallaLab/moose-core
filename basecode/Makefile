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
	consts.o	\
	Element.o	\
	DataHandler.o \
	AnyDimGlobalHandler.o \
	ZeroDimGlobalHandler.o \
	OneDimGlobalHandler.o \
	DataHandlerWrapper.o \
	AnyDimHandler.o \
	ZeroDimHandler.o \
	OneDimHandler.o \
	FieldDataHandlerBase.o \
	MsgDataHandler.o \
	Eref.o	\
	Finfo.o	\
	DestFinfo.o	\
	Cinfo.o	\
	SrcFinfo.o \
	ValueFinfo.o \
	SharedFinfo.o \
	FieldElementFinfo.o \
	Id.o \
	ObjId.o \
	Qinfo.o \
	DataId.o \
	SetGet.o \
	OpFunc.o \
	SparseMatrix.o \
	PrepackedBuffer.o \
	FuncBarrier.o \
	doubleEq.o \
	ReduceBase.o \
	main.o	\
	testAsync.o	\

HEADERS = \
	header.h \
	Cinfo.h \
	PrepackedBuffer.h \
	Conv.h \
	DataId.h \
	Dinfo.h \
	DataDimensions.h \
	DataHandler.h \
	AnyDimHandler.h \
	ZeroDimHandler.h \
	OneDimHandler.h \
	AnyDimGlobalHandler.h \
	ZeroDimGlobalHandler.h \
	OneDimGlobalHandler.h \
	FieldDataHandlerBase.h \
	FieldDataHandler.h \
	MsgDataHandler.h \
	Element.h \
	Eref.h \
	Finfo.h \
	DestFinfo.h \
	Id.h \
	ObjId.h \
	../msg/Msg.h \
	OpFuncBase.h \
	OpFunc.h \
	EpFunc.h \
	ProcInfo.h \
	Qinfo.h \
	SrcFinfo.h \
	ValueFinfo.h \
	LookupValueFinfo.h \
	SharedFinfo.h \
	FieldElementFinfo.h \
	MsgFuncBinding.h \
	ReduceBase.h \
	ReduceFinfo.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS) ../shell/Shell.h
Element.o:	DataHandlerWrapper.h
testAsync.o:	SparseMatrix.h SetGet.h ../scheduling/Clock.h ../scheduling/Tick.h ../scheduling/TickPtr.h ../scheduling/TickMgr.h ../biophysics/IntFire.h ../biophysics/SynBase.h
SparseMsg.o:	SparseMatrix.h
SetGet.o:	SetGet.h ../shell/Neutral.h
DataHandlerWrapper.o: DataHandlerWrapper.h
FuncBarrier.o:	FuncBarrier.h
#TestBasecode.o:	Shell.h ShellWrapper.h

.cpp.o:
	$(CXX) $(CXXFLAGS) -I../msg $< -c

$(TARGET):		$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
