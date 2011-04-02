# Makefile
# 
# Author: Subhasis Ray
# Created: 2011-03-10 12:10:32 (+0530)
TARGET = pymoose.o
# CXXFLAGS += -I../basecode -I../msg -I../shell
.PHONY: clean

HEADERS = pymoose.h \
	moosemodule.h \


OBJ = moosemodule.o \
	pymooseutil.o \


$(TARGET): $(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean: 
	-rm -f *.o

