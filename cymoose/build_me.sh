#!/bin/bash -e
export LDFLAGS="-L. -L/usr/lib -L/usr/local/lib -L/usr/lib/mpi/gcc/openmpi/lib/" 
if [ $# -lt 1 ]; then
    CXX="g++" \
    CC="g++" \
    LDFLAGS="-L. -L/usr/lib -L/usr/local/lib -L/usr/lib/mpi/gcc/openmpi/lib/" \
    python ./setup.py build_ext --inplace
    #echo "Copying latest libmoose.so to /usr/local/lib"
    #cp libmoose.so /usr/local/lib/
else
    echo "Just testing"
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:.
python test.py
