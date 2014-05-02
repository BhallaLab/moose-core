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

echo "Beware, O unsuspecting developer!" 
echo "++ This script builds moose using cmake. If you want to use standard "
echo "++ version, run make only."
echo "++ You may like to customize some varibales in Makefile."
echo ""

( cd ./buildMooseUsingCmake && ./build_me.sh )

