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
	muParser.h	\
	muParserBase.h	\
	muParserBytecode.h	\
	muParserCallback.h	\
	muParserDef.h	\
	muParserError.h	\
	muParserFixes.h	\
	muParserStack.h	\
	muParserToken.h	\
	muParserTokenReader.h		\
	muParserTemplateMagic.h	\


default: $(TARGET)

$(OBJ): $(HEADERS)

.cpp.o:
	$(CXX) $(CXXFLAGS) -I. $< -c

$(TARGET):	$(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*

