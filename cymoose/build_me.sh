#!/bin/bash -e
rm -rf *.c *.cpp
export LDFLAGS="-L. -L/usr/lib -L/usr/local/lib -L/usr/lib/mpi/gcc/openmpi/lib/" 
if [ $# -lt 1 ]; then
    CXX="clang++" \
    CC="clang++" \
    LDFLAGS="-L. -L/usr/lib -L/usr/local/lib -L/usr/lib/mpi/gcc/openmpi/lib/" \
    python ./setup.py build_ext --inplace
else
    echo "Just testing"
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
python test.py
