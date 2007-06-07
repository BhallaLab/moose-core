TARGET = pymoose
OBJFILES = ../basecode/basecode.o ../element/element.o ../shell/shell.o ../builtins/builtins.o ../biophysics/biophysics.o ../scheduling/scheduling.o PyMooseBase.o PyMooseContext.o Compartment.o Neutral.o PyMooseUtil.o HHChannel.o HHGate.o Interpol.o CaConc.o SpikeGen.o SynChan.o Table.o Nernst.o Tick.o ClockJob.o Enzyme.o  KineticHub.o Kintegrator.o Molecule.o  Reaction.o Stoich.o
#  ../kinetics/SparseMatrix.o
CFLAGS = -ggdb -O0 -Wall -fPIC -DDO_UNIT_TESTS
HEADERS = PyMooseUtil.h PyMooseContext.h PyMooseBase.h  Compartment.h Neutral.h HHChannel.h HHGate.h Interpol.h CaConc.h SpikeGen.h SynChan.h Table.h Nernst.h Tick.h ClockJob.h TableIterator.h Enzyme.h KineticHub.h Kintegrator.h Molecule.h Reaction.h Stoich.h 
#../kinetics/SparseMatrix.h
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
	-rm -f *.o moose_wrap.cxx *.so moose.py test_swig *.pyc
