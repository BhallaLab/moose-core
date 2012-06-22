# Makefile
# 
# Author: Subhasis Ray
# Created: 2011-03-10 12:10:32 (+0530)
TARGET = _pymoose.o
# CXXFLAGS += -I../basecode -I../msg -I../shell
.PHONY: clean

HEADERS = moosemodule.h


OBJ = moosemodule.o

$(TARGET): moosemodule.cpp $(OBJ) $(HEADERS) ../basecode/main.cpp ../utility/_utility.o ../basecode/_basecode_pymoose.o
	$(LD) -r -o $(TARGET) $(OBJ)

clean: 
	-rm -f *.o

