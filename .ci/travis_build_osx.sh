#!/bin/bash -
#===============================================================================
#
#          FILE: travis_build_linux.sh
#
#         USAGE: ./travis_build_linux.sh
#
#   DESCRIPTION:  Build  on linux environment.
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: Dilawar Singh (), dilawars@ncbs.res.in
#  ORGANIZATION: NCBS Bangalore
#       CREATED: 01/02/2017 10:11:46 AM
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error
set -e

# NOTE: On travis, don't enable -j`nproc` option. It may not compile properly.
NPROC=$(nproc)
(
    # Make sure not to pick up python from /opt.
    PATH=/usr/local/bin:/usr/bin:$PATH

    PYTHON3=$(which python3)

    # Get pylint
    $PYTHON3 -m pip install pylint --user
    $PYTHON3 -m pip install python-libsbml --user
    $PYTHON3 -m pip install pyneuroml --user

    mkdir -p _GSL_BUILD && cd _GSL_BUILD \
        && cmake -DDEBUG=ON \
        -DPYTHON_EXECUTABLE=$PYTHON3 \
        ..
    make pylint -j$NPROC
    make -j$NPROC && ctest --output-on-failure -$NPROC

    cd .. # Now with boost.
    mkdir -p _BOOST_BUILD && cd _BOOST_BUILD \
        && cmake -DWITH_BOOST_ODE=ON -DDEBUG=ON \
        -DPYTHON_EXECUTABLE=`which python3` ..

    make -j$NPROC && ctest -j$NPROC --output-on-failure 
)
set +e
