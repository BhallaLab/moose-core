

SUBDIRS := basecode utility utility/randnum element builtins biophysics kinetics scheduling shell genesis_parser maindir

CXX = g++
LD = ld

OBJDIR=bin

SOURCES :=
HEADERS := 
OBJECTS = $(subst .cpp,.o,$(SOURCES))
DEPENDENCIES = $(subst .cpp,.d,$(SOURCES))
EXTRA_CLEAN := 
INCLUDE_DIRS := . basecode external/include utility utility/randnum element builtins biophysics kinetics scheduling shell genesis_parser maindir

CXXFLAGS += $(addprefix -I,$(INCLUDE_DIRS)) -DYYMALLOC -DYYFREE -DYYSTYPE_IS_DECLARED -DUSE_GENESIS_PARSER

VPATH = $(INCLUDE_DIRS)

RM := /bin/rm -f
##### INCLUDE SUB-MAKEFILES #########
all:

include maindir/Makefile.mak

include basecode/Makefile.mak

include utility/Makefile.mak

include utility/randnum/Makefile.mak

include element/Makefile.mak

include builtins/Makefile.mak

include biophysics/Makefile.mak

include kinetics/Makefile.mak

include scheduling/Makefile.mak

include shell/Makefile.mak

include genesis_parser/Makefile.mak




##### END OF INCLUDES #########

.PHONY: all

all: moose

moose: $(OBJECTS) $(DEPENDENCIES)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS)

# .PHONY: OBJLIBS			

# OBJLIBS: $(OBJLIBS)

.PHONY: clean

clean:
	$(RM) $(OBJECTS) $(OBJLIBS) $(DEPENDENCIES) $(EXTRA_CLEAN)

include $(DEPENDENCIES)

%.d: %.cpp
	$(CXX) $(CXXFLAGS) -M $< > $@
