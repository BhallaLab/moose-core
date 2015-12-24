# Makefile for including muParser into moose.
# Author: Subhasis Ray
#
TARGET = _muparser.o

OBJ = \
	muParser.o	\
	muParserBase.o	\
	muParserTokenReader.o	\
	muParserError.o	\
	muParserCallback.o	\
	muParserBytecode.o

HEADERS = \
	include/muParser.h	\
	include/muParserBase.h	\
	include/muParserBytecode.h	\
	include/muParserCallback.h	\
	include/muParserDef.h	\
	include/muParserError.h	\
	include/muParserFixes.h	\
	include/muParserStack.h	\
	include/muParserToken.h	\
	include/muParserTokenReader.h		\
	include/muParserTemplateMagic.h	\


default: $(TARGET)

$(OBJ): $(HEADERS)

%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -Iinclude $< -c

$(TARGET):	$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*

