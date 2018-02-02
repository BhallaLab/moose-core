FROM quay.io/pypa/manylinux1_x86_64
MAINTAINER Dilawar Singh <dilawar.s.rajput@gmail.com>

# If you are behind proxy,  uncomment the following lines with appropriate
# values. Otherwise comment them out.
ENV http_proxy http://proxy.ncbs.res.in:3128
ENV https_proxy http://proxy.ncbs.res.in:3128

RUN yum update
RUN yum install -y cmake28 && ln -s /usr/lib/cmake28 /usr/local/bin/cmake
RUN yum install -y wget 
RUN wget --no-check-certificate ftp://ftp.gnu.org/gnu/gsl/gsl-2.4.tar.gz 
RUN tar xvf gsl-2.4.tar.gz && cd gsl-2.4 && ./configure && make -j2 &&  make install && cd 
RUN yum install -y git
RUN git clone --depth 10 https://github.com/BhallaLab/moose-core 
RUN cd moose-core && git pull && cd wheels && ./build_wheels.sh
RUN <<EOR
set -e
set -x

# Clone git or update.
if [ ! -d /tmp/moose-core ]; then
    git clone https://github.com/BhallaLab/moose-core --depth 10 /tmp/moose-core
else
    cd /tmp/moose-core && git pull && cd -
fi

# Try to link statically.
GSL_STATIC_LIBS=/usr/local/lib/libgsl.a

WHEELHOUSE=$HOME/wheelhouse
mkdir -p $WHEELHOUSE
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
        PYTHON=$(ls $PYDIR/bin/python?.?)
        $PYTHON -m pip install numpy
        cmake -DPYTHON_EXECUTABLE=$PYTHON  \
            -DGSL_STATIC_LIBRARIES=$GSL_STATIC_LIBS \
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
EOR
