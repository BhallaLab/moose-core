#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU Lesser General Public License version 2.1
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/
# $Id: $
#
# $Log: $
#

TARGET = _basecode.o


#  CXXFLAGS += -I.. -I../kinetics -I../biophysics

OBJ = \
	consts.o	\
	Element.o	\
	DataElement.o	\
	GlobalDataElement.o	\
	LocalDataElement.o	\
	Eref.o	\
	Finfo.o	\
	DestFinfo.o	\
	Cinfo.o	\
	SrcFinfo.o \
	ValueFinfo.o \
	SharedFinfo.o \
	FieldElementFinfo.o \
	FieldElement.o \
	Id.o \
	ObjId.o \
	SetGet.o \
	OpFuncBase.o \
	EpFunc.o \
	HopFunc.o \
	SparseMatrix.o \
	doubleEq.o \
	testAsync.o	\
	main.o	\

HEADERS = \
	header.h \
	Cinfo.h \
	Conv.h \
	Dinfo.h \
	MsgDigest.h \
	Element.h \
	DataElement.h \
	GlobalDataElement.h \
	LocalDataElement.h \
	Eref.h \
	Finfo.h \
	DestFinfo.h \
	Id.h \
	ObjId.h \
	../msg/Msg.h \
	OpFuncBase.h \
	OpFunc.h \
	EpFunc.h \
	HopFunc.h \
	ProcInfo.h \
	SrcFinfo.h \
	ValueFinfo.h \
	LookupValueFinfo.h \
	LookupElementValueFinfo.h \
	SharedFinfo.h \
	FieldElementFinfo.h \
	FieldElement.h \
	MsgFuncBinding.h \

default: $(TARGET)

$(OBJ)	: $(HEADERS) ../shell/Shell.h
Element.o:	FuncOrder.h
testAsync.o:	SparseMatrix.h SetGet.h ../scheduling/Clock.h ../biophysics/IntFire.h ../biophysics/SpikeRingBuffer.h ../biophysics/SynHandler.h
SparseMsg.o:	SparseMatrix.h
SetGet.o:	SetGet.h ../shell/Neutral.h
HopFunc.o:	HopFunc.h

.cpp.o:
	$(CXX) $(CXXFLAGS) -I../msg $< -c

.PHONY: pymoose

pymoose: CXXFLAGS += -DPYMOOSE
pymoose: OBJ := $(filter-out main.o,$(OBJ))
pymoose: OBJ += main_pymoose.o
pymoose: _basecode_pymoose.o main_pymoose.o
_basecode_pymoose.o: $(OBJ) main_pymoose.o
	$(LD) -r -o _basecode_pymoose.o $(OBJ)

main_pymoose.o: main.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) -I../msg $<
	@echo "Compiled version of main file for pymoose."

$(TARGET):		$(OBJ) $(HEADERS) 
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
