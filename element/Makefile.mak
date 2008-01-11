LOCAL_DIR := element

LOCAL_SRCS := \
	Neutral.cpp	\
	Wildcard.cpp	\

LOCAL_HEADERS := $(subst .cpp,.h,$(LOCAL_SRCS))

SOURCES += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_SRCS))

HEADERS += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_HEADERS))
