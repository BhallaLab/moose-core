CFLAGS += -I..
# the line below are for easy inclusion of libxml++
#CFLAGS += $(shell pkg-config libxml++-2.6 --cflags) 

TARGET = utility.o

OBJ = \
	ArgParser.o	\
	PathUtility.o	\
	StringUtil.o	\
	Property.o	\
	SparseMatrix.o	\

HEADERS = \
	ArgParser.h	\
	PathUtility.h	\
	Property.h	\
	StringUtil.h	\
	utility.h	\
	SparseMatrix.h

default: $(TARGET)

#$(OBJ): $(HEADERS)

ArgParser.o : ArgParser.h
PathUtility.o : PathUtility.h
StringUtil.o : StringUtil.h
Property.o : Property.h
SparseMatrix.o : SparseMatrix.h

.cpp.o:
	$(CXX) $(CFLAGS)  $< -c

$(TARGET): $(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS) 
clean:
	-rm -f *.o $(TARGET) core core.*
