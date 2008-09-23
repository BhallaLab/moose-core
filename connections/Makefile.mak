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

LOCAL_DIR := connections

LOCAL_SRCS := \
	ConnTainer.cpp	\
	SimpleConn.cpp	\
	One2AllConn.cpp	\
	All2OneConn.cpp	\
	Many2ManyConn.cpp	\
	One2OneMapConn.cpp	\
	TraverseMsgConn.cpp	\
	TraverseDestConn.cpp	\

LOCAL_HEADERS := $(subst .cpp,.h,$(LOCAL_SRCS))

SOURCES += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_SRCS))

HEADERS += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_HEADERS))
