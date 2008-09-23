#$/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2007 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************$/
#
# Here we do extra dependencies to get SWIG to compile. The
# final stage of compilation and assorted flags are to be set
# in the moose root Makefile.
#

LOCAL_DIR := shell

LOCAL_SRCS := \
	Shell.cpp	\
	ReadCell.cpp \
	SimDump.cpp \
	LoadTab.cpp \


$(LOCAL_DIR)$/Shell.o: $(LOCAL_DIR)$/Shell.h \
    randnum$/Probability.h \
    randnum$/Uniform.h \
    randnum$/Exponential.h \
    randnum$/Normal.h  \

$(LOCAL_DIR)$/ReadCell.o: $(LOCAL_DIR)$/ReadCell.h $(LOCAL_DIR)$/Shell.h

LOCAL_HEADERS :=  \
	Shell.h	\
	ReadCell.h \
	SimDump.h \

SOURCES += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_SRCS))

HEADERS += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_HEADERS))
