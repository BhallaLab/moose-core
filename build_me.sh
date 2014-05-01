#!/bin/bash
set -e 
#function runPythonTest
#{
#    ( cd python/moose && python test.py )
#}
#
#function runTest 
#{
#    ( echo "q" | ./moose )
#    runPythonTest
#}
#
#function makeMoose
#{
#    make BUILD="$1"
#}
#
#function cleanCode 
#{
#    if [ $# -eq 1 ]; then
#        cd "$1" && make clean
#    else
#        make clean
#    fi
#}
#
#if [ "$1" = "c" ]; then
#    echo "Cleaning $2 and building moose again"
#    cleanCode "$2"
#    makeMoose developer
#    runTest 
#elif [ "$1" = "t" ]; then
#    echo "Running tests"
#    runTest
#    exit
#else
#    makeMoose developer
#    #runTest
#    ( cd tests && ./test_hsolve_in_python.sh )
#    #echo "Running python test only"
#    #runPythonTest
#fi
echo "WARNING: This script builds using cmake. If you want to use Makefile
version, run make only. You might like to change some varibales in Makefile."

( cd ./buildMooseUsingCmake && ./build_me.sh )

