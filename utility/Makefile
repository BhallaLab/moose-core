CXXFLAGS += -I..
# the line below are for easy inclusion of libxml++
#CXXFLAGS += $(shell pkg-config libxml++-2.6 --cflags) 

TARGET = utility.o

OBJ = \
	strutil.o	\
	types.o 	\
	setupenv.o	\


HEADERS = \
	strutil.h \

default: $(TARGET)

strutil.o: strutil.h

.cpp.o:
	$(CXX) $(CXXFLAGS) -I.. $< -c

$(TARGET): $(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) 
clean:
	-rm -f *.o $(TARGET) core core.*
