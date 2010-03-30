CXXFLAGS += -I..
# the line below are for easy inclusion of libxml++
#CXXFLAGS += $(shell pkg-config libxml++-2.6 --CXXFLAGS) 

TARGET = randnum.o

OBJ = \
	mt19937ar.o	\
	Binomial.o	\
	Normal.o	\
	Exponential.o	\
	Poisson.o	\
	Gamma.o		\
	RandGenerator.o	\
	NormalRng.o	\
	PoissonRng.o	\
	BinomialRng.o	\
	ExponentialRng.o	\
	GammaRng.o	\
	UniformRng.o	\
	Uniform.o	\

HEADERS = \
	Probability.h	\
	Binomial.h	\
	Normal.h	\
	Exponential.h	\
	Poisson.h	\
	Gamma.h		\
	randnum.h	\
	RandGenerator.h	\
	NormalRng.h	\
	PoissonRng.h	\
	BinomialRng.h	\
	ExponentialRng.h	\
	GammaRng.h	\
	UniformRng.h	\
	Uniform.h	\
	../basecode/header.h	\
	../basecode/Element.h	\
	../connections/Conn.h	\

default: $(TARGET)

.cpp.o:
	$(CXX) $(CXXFLAGS) $< -c
#	$(CXX) $(CXXFLAGS) $(shell pkg-config libxml++-2.6 --CXXFLAGS) $< -c

$(TARGET): $(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ)

clean:
	-rm -f *.o $(TARGET) core core.*
