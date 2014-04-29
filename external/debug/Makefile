TARGET = _debug.o

OBJ = print_function.o

HEADER = print_function.h

default: $(TARGET)

$(OBJ) : $(HEADER)
print_function.o: print_function.h

.cpp.o:
	$(CXX) $(CXXFLAGS) $< -c

$(TARGET): $(OBJ) $(HEADER)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -rf *.o $(TARGET)
