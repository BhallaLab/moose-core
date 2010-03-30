CXXFLAGS += -I..
# the line below are for easy inclusion of libxml++
#CXXFLAGS += $(shell pkg-config libxml++-2.6 --cflags) 

TARGET = utility.o

OBJ = \
	ArgParser.o	\
	PathUtility.o	\
	NumUtil.o	\
	StringUtil.o	\
	Property.o	\
	SparseMatrix.o	\
	Helpless.o	\
	help.o		\

HEADERS = \
	ArgParser.h	\
	PathUtility.h	\
	NumUtil.h	\
	Property.h	\
	StringUtil.h	\
	utility.h	\
	SparseMatrix.h

default: $(TARGET)

#$(OBJ): $(HEADERS)

ArgParser.o : ArgParser.h
PathUtility.o : PathUtility.h
NumUtil.o: NumUtil.h
StringUtil.o : StringUtil.h
Property.o : Property.h
SparseMatrix.o : SparseMatrix.h

.cpp.o:
	$(CXX) $(CXXFLAGS) -I.. $< -c

$(TARGET): $(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) 
clean:
	-rm -f *.o $(TARGET) core core.*
