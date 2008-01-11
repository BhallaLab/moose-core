#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

LOCAL_DIR := biophysics

LOCAL_SRCS := \
	Compartment.cpp	\
	SymCompartment.cpp	\
	HHChannel.cpp	\
	Mg_block.cpp	\
	HHGate.cpp	\
	CaConc.cpp	\
	Nernst.cpp	\
	SpikeGen.cpp	\
	SynChan.cpp	\
	HSolveStruct.cpp	\
	HSolveBase.cpp	\
	HSolve.cpp \
	NeuroScanBase.cpp \
	NeuroScan.cpp \
	NeuroHub.cpp \
	Cell.cpp	\
	TestBiophysics.cpp	\
	BinSynchan.cpp	\
	StochSynchan.cpp	

$(LOCAL_DIR)/Compartment.o: $(LOCAL_DIR)/Compartment.h basecode/Ftype2.h
$(LOCAL_DIR)/SymCompartment.o: $(LOCAL_DIR)/Compartment.h $(LOCAL_DIR)/SymCompartment.h basecode/Ftype2.h
$(LOCAL_DIR)/HHChannel.o: $(LOCAL_DIR)/HHChannel.h basecode/Ftype2.h basecode/Ftype3.h
$(LOCAL_DIR)/Mg_block.o: $(LOCAL_DIR)/Mg_block.h basecode/Ftype2.h basecode/Ftype3.h
$(LOCAL_DIR)/HHGate.o: $(LOCAL_DIR)/HHGate.h builtins/Interpol.h
$(LOCAL_DIR)/BinSynInfo.h: utility/randnum/BinomialRng.h

LOCAL_HEADERS := 	\
	Compartment.h	\
	SymCompartment.h	\
	HHChannel.h	\
	Mg_block.h	\
	HHGate.h	\
	CaConc.h	\
	Nernst.h	\
	SpikeGen.h	\
	SynChan.h	\
	HSolveStruct.h	\
	HSolveBase.h	\
	HSolve.h \
	NeuroScanBase.h \
	NeuroScan.h \
	NeuroHub.h \
	Cell.h	\
	BinSynchan.h	\
	StochSynchan.h	


SOURCES += $(addprefix $(LOCAL_DIR)/, $(LOCAL_SRCS))

HEADERS += $(addprefix $(LOCAL_DIR)/, $(LOCAL_HEADERS))

