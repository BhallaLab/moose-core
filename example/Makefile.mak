#$/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************$/

LOCAL_DIR := example

LOCAL_SRCS := \
	Average.cpp	\

LOCAL_HEADERS := Average.h


SOURCES += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_SRCS))
HEADERS += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_HEADERS))
