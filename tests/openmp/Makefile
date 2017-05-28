MKFILE_PATH := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))

all : test_with_openmp test_without_openmp 
	echo "Test is done"

test_without_openmp : ./test_ksolve.py 
	PYTHONPATH='' python $<

test_with_openmp : ./test_ksolve.py 
	PYTHONPATH=$(MKFILE_PATH)/../../_build/python python $<

