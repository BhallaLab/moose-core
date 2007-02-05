TARGET = shell.so

OBJ = \
	Shell.o \
	main.o \

default: $(TARGET)

Shell.o: Shell.h
Swig_wrap.o: Swig.h Swig_wrap.cxx


Swig_wrap.cxx:
	swig -c++ -python Swig.i

Swig_wrap.o:
	g++ -fpic -c Swig_wrap.cxx -I/usr/include/python2.4

.cpp.o:
	g++ -fpic $< -c

$(TARGET):	$(OBJ) Swig_wrap.o
	g++ -shared Swig_wrap.o $(OBJ) -o _$(TARGET)
	
noswig: $(OBJ)
	g++ $(OBJ) -o noswig

clean:
	-rm -f *.o Swig_wrap.cxx *.so

