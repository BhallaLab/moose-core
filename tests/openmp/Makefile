MKFILE_PATH := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))

all : test_with_openmp test_without_openmp 
	echo "Test is done"

test_without_openmp : ./test_ksolve.py 
	@echo "Without OpenMP"
	PYTHONPATH='' python $<

test_with_openmp : ./test_ksolve.py 
	@echo "With OpenMP"
	#PYTHONPATH=$(MKFILE_PATH)/../../_build/python python $<
	OMP_NUM_THREADS=1 PYTHONPATH=$(MKFILE_PATH)/../../_build/python python $<

