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
# Normally this Makefile is called from the master Makefile to make the
# default SLI.o
#
# If you want to rebuild the parser from scratch, you need to do
# 'make parser' in this directory. You will need bison, flex, and sed.
#

TARGET = SLI.o
CXXFLAGS += -I..

ifneq (,$(findstring -DCRL_MPI,$(CXXFLAGS)))
OBJ = \
	GenesisParser.o \
	GenesisParserWrapper.o \
	ParGenesisParser.o \
	GenesisParser.tab.o \
	symtab.o \
	eval.o \
	exec_fork.o \
	parse.o \
	getopt.o \
	script.o 
else
OBJ = \
	GenesisParser.o \
	GenesisParserWrapper.o \
	GenesisParser.tab.o \
	symtab.o \
	eval.o \
	exec_fork.o \
	parse.o \
	getopt.o \
	script.o 
endif

HEADERS = \
	../basecode/header.h \
	../connections/Conn.h \
	../basecode/Element.h \
	../utility/utility.h	\

default: $(TARGET)


$(OBJ)	: GenesisParser.yy.cpp $(HEADERS)
GenesisParser.o:			GenesisParser.h
GenesisParserWrapper.o:		GenesisParser.h GenesisParserWrapper.h ../shell/Shell.h
ifneq (,$(findstring -DCRL_MPI,$(CXXFLAGS)))
ParGenesisParser.o:		GenesisParserWrapper.h ParGenesisParser.h ../shell/Shell.h
endif
GenesisParser.yy.o:			GenesisParser.yy.cpp
GenesisParser.yy.cpp:			GenesisParser.yy.l
	flex++ -+ -oGenesisParser.yy.cpp GenesisParser.yy.l
	sed -e "s/^    #define yyFlexLexer yyFlexLexer/\/\/ Deleted    #define yyFlexLexer yyFlexLexer/" GenesisParser.yy.cpp > temp
	sed -e "s/^extern \"C\" int isatty (int );/\/\/ Deleted extern C int isatty/" temp > GenesisParser.yy.cpp
GenesisParser.tab.o:	GenesisParser.tab.cpp
GenesisParser.tab.cpp:	GenesisParser.ypp GenesisParser.yy.cpp
	bison -d GenesisParser.ypp
	sed -e "/^extern\ YYSTYPE/D" GenesisParser.tab.hpp > GenesisParser.tab.h 
	sed -e "s/^yyparse/myFlexLexer::yyparse/" GenesisParser.tab.cpp > temp
	sed -e "s/YYSTACK_FREE (yyss);/    ;\/\/ Deleted YYSTACK_FREE( yyss);/" temp > temp2
	sed -e "s/^YY_DECL_VARIABLES/\/\/ Shifted YY_DECL_VARIABLES to inside myFlexLexer class/" temp2 > GenesisParser.tab.cpp

GenesisParser.tab.o GenesisParser.yy.o GenesisParser.o: GenesisParser.yy.cpp GenesisParser.h GenesisParser.tab.h

# GenesisParser.tab.h GenesisParser.tab.cpp: GenesisParser.ypp
# 	bison -d GenesisParser.ypp
# 	sed -e "/^extern\ YYSTYPE/D" GenesisParser.tab.hpp > GenesisParser.tab.h
# #/bin/mv GenesisParser.tab.hpp GenesisParser.tab.h
# 	sed -e "s/^yyparse/myFlexLexer::yyparse/" GenesisParser.tab.cpp > temp
# 	sed -e "s/YYSTACK_FREE (yyss);/    ;\/\/ Deleted YYSTACK_FREE( yyss);/" temp > temp2
# 	sed -e "s/^YY_DECL_VARIABLES/\/\/ Shifted YY_DECL_VARIABLES to inside myFlexLexer class/" temp2 > GenesisParser.tab.cpp 


# 	sed -e "s/^    #define yyFlexLexer yyFlexLexer/\/\/ Deleted    #define yyFlexLexer yyFlexLexer/" GenesisParser.yy.cpp > temp
# 	sed -e "s/^extern \"C\" int isatty (int );/\/\/ Deleted extern C int isatty/" temp > GenesisParser.yy.cpp

.cpp.o:
	$(CXX) $(CXXFLAGS) -DYYMALLOC -DYYFREE -DYYSTYPE_IS_DECLARED -I../basecode -I../external/include $< -c

$(TARGET):		$(OBJ) $(HEADERS)
		$(LD) -r -o $(TARGET) $(OBJ) 
#		$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.* GenesisParser.tab.o GenesisParser.yy.o
# 	-rm -f *.o $(TARGET) core core.* GenesisParser.tab.* GenesisParser.yy.o GenesisParser.yy.cpp temp temp2
