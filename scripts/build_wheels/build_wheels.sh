#!/bin/bash

# NOTE: Use this script only on manylinux docker image. This builds pymoose
# wheels.
echo "[INFO] Make sure to use it on manylinux docker image else it may \
    not work"

set -e
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

MOOSE_SOURCE_DIR=/tmp/moose

# Clone git or update.
if [ ! -d $MOOSE_SOURCE_DIR ]; then
    git clone -b chennapoda https://github.com/BhallaLab/moose-core --depth 10 $MOOSE_SOURCE_DIR
else
    git pull
fi

# This library much be available as static built.
GSL_STATIC_LIBS="/usr/local/lib/libgsl.a;/usr/local/lib/libgslcblas.a"
CMAKE=/usr/bin/cmake28

WHEELHOUSE=$HOME/wheelhouse
mkdir -p $WHEELHOUSE
for PYDIR in /opt/python/cp27-cp27m/ /opt/python/cp34-cp34m/ /opt/python/cp36-cp36m/; do
    PYVER=$(basename $PYDIR)
    mkdir -p $PYVER
    (
        cd $PYVER
        echo "Building using $PYDIR in $PYVER"
        PYTHON=$(ls $PYDIR/bin/python?.?)
        $PYTHON -m pip install numpy
        $CMAKE -DPYTHON_EXECUTABLE=$PYTHON  \
            -DGSL_STATIC_LIBRARIES=$GSL_STATIC_LIBS \
            -DMOOSE_VERSION="3.2rc1" ${MOOSE_SOURCE_DIR}
        make -j4

        # Now build bdist_wheel
        cd python
        cp setup.cmake.py setup.py
        $PYDIR/bin/pip wheel . -w $WHEELHOUSE
    )
done

# now check the wheels.
for whl in $WHEELHOUSE/*.whl; do
    #auditwheel repair "$whl" -w $WHEELHOUSE
    auditwheel show "$whl"
done
ls -lh $WHEELHOUSE/*.whl
