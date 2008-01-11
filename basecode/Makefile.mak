LOCAL_DIR := basecode

LOCAL_SRCS := \
	Element.cpp	\
	Id.cpp	\
	IdManager.cpp	\
	Conn.cpp	\
	RecvFunc.cpp \
	SimpleElement.cpp	\
	ArrayElement.cpp	\
	ArrayWrapperElement.cpp	\
	Copy.cpp \
	send.cpp \
	SrcFinfo.cpp \
	DestFinfo.cpp \
	ValueFinfo.cpp \
	LookupFinfo.cpp \
	ThisFinfo.cpp \
	DeletionMarkerFinfo.cpp \
	GlobalMarkerFinfo.cpp \
	Cinfo.cpp \
	DynamicFinfo.cpp \
	ExtFieldFinfo.cpp \
	UnitTests.cpp \
	TestBasecode.cpp \
	strconv.cpp \
	DerivedFtype.cpp \
	SharedFtype.cpp \
	SharedFinfo.cpp \
	SolveFinfo.cpp \
	setget.cpp \
	filecheck.cpp \
	FunctionData.cpp \
	ParallelDummy.cpp \
	Class.cpp	\

LOCAL_HEADERS := \
	header.h \
	RecvFunc.h \
	Conn.h \
	Ftype.h \
	FunctionData.h \
	Finfo.h \
	Element.h \
	Id.h \
	Class.h \
	Element.h	\
	Id.h	\
	IdManager.h	\
	Conn.h	\
	RecvFunc.h \
	SimpleElement.h	\
	ArrayElement.h	\
	ArrayWrapperElement.h	\
	send.h \
	SrcFinfo.h \
	DestFinfo.h \
	ValueFinfo.h \
	LookupFinfo.h \
	ThisFinfo.h \
	DeletionMarkerFinfo.h \
	GlobalMarkerFinfo.h \
	Cinfo.h \
	DynamicFinfo.h \
	ExtFieldFinfo.h \
	DerivedFtype.h \
	SharedFtype.h \
	SharedFinfo.h \
	SolveFinfo.h \
	setget.h \
	filecheck.h \
	FunctionData.h \
	Class.h	\

SOURCES += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_SRCS))

HEADERS += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_HEADERS))
