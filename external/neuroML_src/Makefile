TARGET	   = neuroml
INCLUDEDIR = /usr/include/libxml2/
CXX        = g++
CXXFLAGS   = -g -Wall 
LDFLAGS    = 
LIBS       = -lxml2 -L/usr/lib/

# Get the processor architecture - i686 or x86_64
MACHINE?=i686 
ifneq ($(OSTYPE),win32)
MACHINE=$(shell uname -m)
endif

# For 64 bit Linux systems add paths to 64 bit libraries 
ifeq ($(OSTYPE),Linux)
ifeq ($(MACHINE),x86_64)
LIBS=-L/lib64 -L/usr/lib64 $(LIBS) 
endif
endif

TYPE ?= static
OBJ = \
	NBase.o		\
	NCell.o		\
	Segment.o	\
	Channel.o	\
	NCable.o	\
    SynChannel.o    \
    IonPool.o	\

HEADERS = \
	NBase.h		\
	NCell.h		\
	Segment.h      \
	Channel.h	\
	NCable.h 	\
    SynChannel.h    \
    IonPool.h	\
	
default: $(TARGET)

$(OBJ): $(HEADERS)
NBase.o:	NBase.h
NCell.o:	Segment.h NCable.h Channel.h  SynChannel.h IonPool.h NCell.h
Segment.o: 	Segment.h 
Channel.o:	NCell.h Channel.h
Cable.o:	NCable.h
SynChannel.o:   NCell.h SynChannel.h 
IonPool.o:	NCell.h IonPool.h

.cpp.o:
	$(CXX) $(CXXFLAGS) -I$(INCLUDEDIR) $< -o $@ -c

ifeq ($(TYPE),static)
$(TARGET): $(OBJ) $(HEADERS)
	$(LD) -r -o neuroml.o $(OBJ) 
	ar rcs libneuroml.a neuroml.o
	@echo "Created static library"
endif
ifeq ($(TYPE),dynamic)
CXXFLAGS += -fPIC
ifeq ($(OSTYPE),Darwin)
LDFLAGS += -bundle -fno-common
else
LDFLAGS  += -shared
endif

$(TARGET): $(OBJ) $(HEADERS)
	$(CXX)  $(CXXFLAGS) -I$(INCLUDEDIR) $(LDFLAGS) $(OBJ) $(LIBS) -o libneuroml.so
	@echo "Created dynamic library"
endif

clean:
	-rm -f *.o *.a *.so core core.* 
