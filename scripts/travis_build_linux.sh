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

(
    cd ..
    make 
    ## CMAKE based flow
    mkdir -p _GSL_BUILD && cd _GSL_BUILD && cmake -DDEBUG=ON -DPYTHON_EXECUTABLE=`which python` ..
    make && ctest --output-on-failure
    cd .. # Now with boost.
    mkdir -p _BOOST_BUILD && cd _BOOST_BUILD && cmake -DWITH_BOOST=ON -DDEBUG=ON -DPYTHON_EXECUTABLE=`which python` ..
    make && ctest --output-on-failure
    cd .. 
    echo "Python3 support. Removed python2-networkx and install python3"
    sudo apt-get remove -qq python-networkx 
    sudo apt-get install -qq python3-networkx
    mkdir -p _GSL_BUILD2 && cd _GSL_BUILD2 && cmake -DDEBUG=ON -DPYTHON_EXECUTABLE=`which python3` ..
    make && ctest --output-on-failure
    cd .. # Now with BOOST and python3
    mkdir -p _BOOST_BUILD2 && cd _BOOST_BUILD2 && cmake -DWITH_BOOST=ON -DDEBUG=ON -DPYTHON_EXECUTABLE=`which python3` ..
    make && ctest --output-on-failure
)
