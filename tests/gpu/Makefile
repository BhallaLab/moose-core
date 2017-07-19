MKFILE_PATH := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))

all : test_with_cuda test_without_cuda 
	echo "Test is done"

test_without_cuda : ./test_ksolve.py 
	@echo "Without Cuda"
	PYTHONPATH='' python $<

test_with_cuda : ./test_ksolve.py 
	@echo "With Cuda"
	PYTHONPATH=$(MKFILE_PATH)/../../_build/python python $<
	#PYTHONPATH=$(MKFILE_PATH)/../../_build/python python_in_gdb.sh $<

        
