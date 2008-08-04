LOCAL_DIR := utility

LOCAL_SRCS := \
	ArgParser.cpp	\
	PathUtility.cpp	\
	StringUtil.cpp	\
	Property.cpp	\
    SparseMatrix.cpp\

# the line below are for easy inclusion of libxml++
#CFLAGS += $(shell pkg-config libxml++-2.6 --cflags)

LOCAL_HEADERS := \
	ArgParser.h	\
	PathUtility.h	\
	Property.h	\
	StringUtil.h	\
	SparseMatrix.h  \
	utility.h

SOURCES += $(addprefix $(LOCAL_DIR)/, $(LOCAL_SRCS))

HEADERS += $(addprefix $(LOCAL_DIR)/, $(LOCAL_HEADERS))

