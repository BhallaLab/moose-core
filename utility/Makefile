CFLAGS += -I..
# the line below are for easy inclusion of libxml++
#CFLAGS += $(shell pkg-config libxml++-2.6 --cflags) 

TARGET = utility.o

OBJ = \
	ArgParser.o	\
	PathUtility.o	\
	StringUtil.o	\
	Property.o	\

HEADERS = \
	ArgParser.h	\
	PathUtility.h	\
	Property.h	\
	StringUtil.h	\
	utility.h

default: $(TARGET)

.cpp.o:
	$(CXX) $(CFLAGS)  $< -c

$(TARGET): $(OBJ) $(HEADERS) randgen
	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

randgen:
	@(echo cd randnum; cd randnum; $(MAKE); cd ..)
clean:
	@(echo cd randnum; cd randnum; $(MAKE) clean; cd ..)
	-rm -f *.o $(TARGET) core core.*
