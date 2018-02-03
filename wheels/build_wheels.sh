#!/bin/bash
set -e
set -x

# Clone git or update.
if [ ! -d /tmp/moose-core ]; then
    git clone -b wheels https://github.com/BhallaLab/moose-core --depth 10 /tmp/moose-core
else
    cd /tmp/moose-core && git pull && cd -
fi

# Try to link statically.
GSL_STATIC_LIBS="/usr/local/lib/libgsl.a;/usr/local/bin/libgslcblas.a"

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
        cmake -DPYTHON_EXECUTABLE=$PYTHON  \
            -DGSL_STATIC_LIBRARIES=$GSL_STATIC_LIBS \
	    -DVERSION_MOOSE=3.2.0 \
            ../..
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
