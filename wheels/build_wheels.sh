#!/bin/bash
set -e

# Clone git or update.
if [ ! -d /tmp/moose-core ]; then
    git clone -b manylinux https://github.com/BhallaLab/moose-core --depth 10 /tmp/moose-core
else
    cd /tmp/moose-core && git pull && cd -
fi

# Try to link statically.
GSL_STATIC_LIBS=/usr/local/lib/libgsl.a

for PYDIR in /opt/python/*; do
    PYVER=$(basename $PYDIR)
    if [[ $PYVER = *"cpython"* ]]; then
        continue
    fi
    if [[ $PYVER = *"cp33"* ]]; then
        continue
    fi
    mkdir -p $PYVER
    (
        cd $PYVER
        echo "Building using $PYDIR in $PYVER"
        PYTHON=$PYDIR/bin/python
        $PYTHON -m pip install numpy
        cmake -DPYTHON_EXECUTABLE=$PYTHON  \
            -DGSL_STATIC_LIBRARIES=$GSL_STATIC_LIBS \
            ../..
        make -j4
    )
done
