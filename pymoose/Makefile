TARGET = pymoose
OBJFILES = ../basecode/basecode.o ../element/element.o ../shell/shell.o ../builtins/builtins.o ../biophysics/biophysics.o PyMooseBase.o PyMooseContext.o Compartment.o Neutral.o PyMooseUtil.o 
CFLAGS = -ggdb -O0 -Wall -fPIC -DDO_UNIT_TESTS
HEADERS = PyMooseUtil.h PyMooseContext.h PyMooseBase.h  Compartment.h Neutral.h 
default: $(TARGET)

test_swig: $(OBJFILES) $(HEADERS) SwigTest.cpp 
	$(CXX) $(CFLAGS) -o test_swig SwigTest.cpp $(OBJFILES)  -I../basecode -I../shell

pymoose: test_swig _moose.so moose.py
_moose.so: moose_wrap.o $(HEADERS) $(OBJFILES)
	g++ -g -shared -o _moose.so moose_wrap.o $(OBJFILES)

moose_wrap.o: moose_wrap.cxx
	g++ -g -c -fPIC moose_wrap.cxx -o moose_wrap.o -I../basecode -I../shell -I/usr/include/python2.4

moose.py moose_wrap.cxx: moose.i $(HEADERS)
	swig -modern -c++ -python moose.i

.cpp.o:
	$(CXX) $(CFLAGS) -I../basecode -I../shell $< -c
clean:
	-rm -f *.o moose_wrap.cxx *.so moose.py test_swig
