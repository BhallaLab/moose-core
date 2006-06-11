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

TARGET = SLI.o

OBJ = \
	GenesisParser.o \
	GenesisParser.tab.o \
	GenesisParserWrapper.o \
	symtab.o \
	eval.o \
	parse.o \
	getopt.o \
	script.o

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


$(OBJ)	: GenesisParser.yy.cpp $(HEADERS)
GenesisParserWrapper.o:		../basecode/Shell.h ../basecode/ShellWrapper.h GenesisParser.h GenesisParserWrapper.h
GenesisParser.o:			../basecode/Shell.h GenesisParser.h
GenesisParser.yy.o:			GenesisParser.yy.l

GenesisParser.yy.cpp: GenesisParser.yy.l
	flex++ -oGenesisParser.yy.cpp GenesisParser.yy.l


GenesisParser.tab.o GenesisParser.yy.o GenesisParser.o: GenesisParser.yy.cpp GenesisParser.h GenesisParser.tab.h

GenesisParser.tab.h GenesisParser.tab.cpp: GenesisParser.ypp
	bison -d GenesisParser.ypp
	sed -e "/^extern\ YYSTYPE/D" GenesisParser.tab.hpp > GenesisParser.tab.h
#/bin/mv GenesisParser.tab.hpp GenesisParser.tab.h
	sed -e "s/^yyparse/myFlexLexer::yyparse/" GenesisParser.tab.cpp > temp
	sed -e "s/^YY_DECL_VARIABLES/\/\/ Shifted YY_DECL_VARIABLES to inside myFlexLexer class/" temp > GenesisParser.tab.cpp 

.cpp.o:
	$(CXX) $(CFLAGS) -DYYSTYPE_IS_DECLARED -I../basecode $< -c

$(TARGET):		$(OBJ) $(HEADERS)
		$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.* GenesisParser.tab.* GenesisParser.yy.o GenesisParser.yy.cpp temp
