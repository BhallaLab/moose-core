##
## Makefile
##  
## Made by subhasis ray
## Login   <subhayantranaa>
##
## Started on  Mon Dec 29 15:52:03 2008 subhasis ray
## Last update Mon Dec 29 15:52:03 2008 subhasis ray
## 

CXXFLAGS += -I.. -I../basecode
TARGET=device.o
OBJ= \
	DiffAmp.o \
	PIDController.o \
	RC.o	\
	PulseGen.o	\
	Efield.o	\


HEADERS= \
	DiffAmp.h \
	PIDController.h \
	RC.h	\
	PulseGen.h	\
	Efield.h	\


default: $(TARGET)
$(OBJ): $(HEADERS)

$(TARGET): $(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
