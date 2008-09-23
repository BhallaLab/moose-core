LOCAL_DIR := basecode

LOCAL_SRCS := \
	Element.cpp	\
    Eref.cpp    \
	Id.cpp	\
	IdManager.cpp	\
	Fid.cpp \
	RecvFunc.cpp \
	Ftype.cpp   \
	Msg.cpp \
	SimpleElement.cpp	\
	ArrayElement.cpp	\
	Copy.cpp \
	Send.cpp \
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
	ParallelDummy.cpp \
	Class.cpp	\
	FuncVec.cpp \

LOCAL_HEADERS := \
	header.h \
	RecvFunc.h \
	Ftype.h \
	Finfo.h \
	Element.h \
	Id.h \
	Class.h \
	Element.h	\
	Eref.h  \
	Id.h	\
	IdManager.h	\
	RecvFunc.h \
	SimpleElement.h	\
	ArrayElement.h	\
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
	Class.h	\

SOURCES += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_SRCS))

HEADERS += $(addprefix $(LOCAL_DIR)$/, $(LOCAL_HEADERS))
