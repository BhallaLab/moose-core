LOCAL_DIR := utility$/randnum
LOCAL_SRCS := \
	mt19937ar.cpp	\
	Binomial.cpp	\
	Normal.cpp	\
	Exponential.cpp	\
	Poisson.cpp	\
	Gamma.cpp		\
	RandGenerator.cpp	\
	NumUtil.cpp	\
	NormalRng.cpp	\
	PoissonRng.cpp	\
	BinomialRng.cpp	\
	ExponentialRng.cpp	\
	GammaRng.cpp	\

LOCAL_HEADERS := \
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
# 	..$/..$/basecode$/header.h	\
# 	..$/..$/basecode$/Element.h	\
# 	..$/..$/basecode$/Conn.h	\

LOCAL_OBJS := $(subst .cpp,.o, $(LOCAL_SRCS))

SOURCES += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_SRCS))

HEADERS += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_HEADERS))
