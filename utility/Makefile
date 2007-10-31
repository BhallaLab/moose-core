CFLAGS += -I..
# the line below are for easy inclusion of libxml++
#CFLAGS += $(shell pkg-config libxml++-2.6 --cflags) 

TARGET = utility.o

OBJ = \
	Configuration.o \
	StringUtil.o	\
	PropertyReader.o \
	mt19937ar.o	\
	Binomial.o	\
	Normal.o

HEADERS = \
	Probability.h	\
	Binomial.h	\
	Normal.h	\
	randnum.h	\
	PropertyReader.h	\
	StringUtil.h	\
	Configuration.h	\
	utility.h

default: $(TARGET)

.cpp.o:
	$(CXX) $(CFLAGS)  $< -c
#	$(CXX) $(CFLAGS) $(shell pkg-config libxml++-2.6 --cflags) $< -c

$(TARGET): $(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
