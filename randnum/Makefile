CFLAGS += -I..
# the line below are for easy inclusion of libxml++
#CFLAGS += $(shell pkg-config libxml++-2.6 --cflags) 

TARGET = randnum.o

OBJ = \
	mt19937ar.o	\
	Binomial.o	\
	Normal.o	\
	Exponential.o	\
	Poisson.o	\
	Gamma.o		\
	RandGenerator.o	\
	NumUtil.o	\
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
	NumUtil.h	\
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
	$(CXX) $(CFLAGS) $< -c
#	$(CXX) $(CFLAGS) $(shell pkg-config libxml++-2.6 --cflags) $< -c

$(TARGET): $(OBJ) $(HEADERS)
	$(LD) -r -o $(TARGET) $(OBJ) $(LIBS)

clean:
	-rm -f *.o $(TARGET) core core.*
