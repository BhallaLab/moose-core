LOCAL_DIR := builtins

LOCAL_SRCS := \
	Interpol.cpp	\
	Table.cpp	\
        TimeTable.cpp   \
        AscFile.cpp     \
        Calculator.cpp


LOCAL_HEADERS := $(subst .cpp,.h,$(LOCAL_SRCS))

SOURCES += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_SRCS))

HEADERS += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_HEADERS))
