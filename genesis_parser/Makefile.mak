#$/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2004 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************$/
# $Id: $
#
# $Log: $
#
# Normally this Makefile is called from the master Makefile to make the
# default SLI.o
#
# If you want to rebuild the parser from scratch, you need to do
# 'make parser' in this directory. You will need bison, flex, and sed.
#

LOCAL_DIR := genesis_parser

LOCAL_SRCS := \
	GenesisParser.cpp \
	GenesisParserWrapper.cpp \
	GenesisParser.tab.cpp \
	symtab.cpp \
	eval.cpp \
	exec_fork.cpp   \
	parse.cpp \
	getopt.cpp \
	script.cpp \

LOCAL_HEADERS := \
	GenesisParser.h \
	GenesisParserWrapper.h \
	GenesisParser.tab.h \
	script.h \


SOURCES += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_SRCS))
HEADERS += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_HEADERS))
