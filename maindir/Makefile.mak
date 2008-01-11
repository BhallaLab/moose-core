LOCAL_DIR := maindir

LOCAL_SRCS := \
	   main.cpp	\
	   init.cpp	\
	   nonblock.cpp	\


LOCAL_HEADERS := \
	      init.h	\


SOURCES += $(addprefix $(LOCAL_DIR)/, $(LOCAL_SRCS))

HEADERS += $(addprefix $(LOCAL_DIR)/, $(LOCAL_HEADERS))
