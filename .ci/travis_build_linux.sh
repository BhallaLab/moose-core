#!/usr/bin/env bash
#
#   DESCRIPTION:  Build  on linux environment.
#
#        AUTHOR: Dilawar Singh (), dilawars@ncbs.res.in
#  ORGANIZATION: NCBS Bangalore
#       CREATED: 01/02/2017 10:11:46 AM

set -e
set -x

BUILDDIR=_build_travis
mkdir -p $BUILDDIR

PYTHON3="/usr/bin/python3"


$PYTHON3 -m pip install pip --upgrade --user
$PYTHON3 -m pip install libNeuroML pyNeuroML python-libsbml --upgrade --user

# sympy is only needed for pretty-priting for one test.
$PYTHON3 -m pip install numpy sympy scipy --upgrade --user   

# pytest requirements.
$PYTHON3 -m pip install -r ./tests/requirements.txt --user
 
NPROC=$(nproc)
MAKE="make -j$NPROC"

unset PYTHONPATH

# Bug: `which python` returns /opt/bin/python* etc on travis. For which numpy
# many not be available. Therefore, it is neccessary to use fixed path for
# python executable.

$PYTHON3 -m compileall -q . 

# Python3 with GSL in debug more.
(
    mkdir -p $BUILDDIR && cd $BUILDDIR && \
        cmake -DPYTHON_EXECUTABLE=$PYTHON3 \
        -DCMAKE_INSTALL_PREFIX=/usr -DDEBUG=ON ..
    $MAKE
    # Run with valgrind to log any memory leak.
    valgrind --leak-check=full ./moose.bin -q -u 

    # Run all tests in debug mode.
    MOOSE_NUM_THREADS=$NPROC ctest -j$NPROC --output-on-failure 


    make install || sudo make install 
    cd /tmp
    $PYTHON3 -c 'import moose;print(moose.__file__);print(moose.version())'
)

# BOOST and python3
(
    mkdir -p $BUILDDIR && cd $BUILDDIR && \
        cmake -DWITH_BOOST_ODE=ON -DPYTHON_EXECUTABLE="$PYTHON3" \
        -DCMAKE_INSTALL_PREFIX=/usr ..
    # Run coverage 
    export MOOSE_NUM_THREADS=3
    make coverage 
)


echo "All done"
