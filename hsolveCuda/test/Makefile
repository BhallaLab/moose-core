CXXFLAGS =	-g -lm -I/usr/include/python2.7

OBJS =	\
 	testMooseCuda.o \

TARGET = testMooseCuda

$(TARGET):	$(OBJS)
		g++ -o $(TARGET) $(OBJS) $(LIBS) -lm -lpython2.7 -L/usr/lib/python2.7/config -L../cudaLibrary 

default: $(TARGET)

%.o: %.cpp
	g++ $(CXXFLAGS) -c -o "$@" "$<"
clean:
	rm -f $(OBJS) $(TARGET)
