CXXFLAGS += -I..
# the line below are for easy inclusion of libxml++
#CXXFLAGS += $(shell pkg-config libxml++-2.6 --CXXFLAGS) 

TARGET = randnum.o

OBJ = \
	mt19937ar.o	\

HEADERS = \
	randnum.h	\

default: $(TARGET)

.cpp.o:
	$(CXX) $(CXXFLAGS) $< -c
#	$(CXX) $(CXXFLAGS) $(shell pkg-config libxml++-2.6 --CXXFLAGS) $< -c

$(TARGET): $(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
