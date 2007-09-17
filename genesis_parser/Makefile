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
	GenesisParserWrapper.o \
	GenesisParser.tab.o \
	symtab.o \
	eval.o \
	parse.o \
	getopt.o \
	script.o \

HEADERS = \
	../basecode/header.h \
	../basecode/Conn.h \
	../basecode/Element.h \

default: $(TARGET)


$(OBJ)	: GenesisParser.yy.cpp $(HEADERS)
GenesisParser.o:			GenesisParser.h
GenesisParserWrapper.o:		GenesisParser.h GenesisParserWrapper.h ../shell/Shell.h
GenesisParser.yy.o:			GenesisParser.yy.cpp
#GenesisParser.yy.o:			GenesisParser.yy.l

# GenesisParser.yy.cpp: GenesisParser.yy.l
# 	flex++ -+ -oGenesisParser.yy.cpp GenesisParser.yy.l


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
	$(CXX) $(CFLAGS) -DYYMALLOC -DYYFREE -DYYSTYPE_IS_DECLARED -I../basecode -I../external/include $< -c

$(TARGET):		$(OBJ) $(HEADERS)
		$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.* GenesisParser.tab.o GenesisParser.yy.o
# 	-rm -f *.o $(TARGET) core core.* GenesisParser.tab.* GenesisParser.yy.o GenesisParser.yy.cpp temp temp2
