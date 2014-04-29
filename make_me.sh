#!/bin/bash

function runPythonTest
{
    ( cd python/moose && python test.py )
}

function runTest 
{
    ( echo "q" | ./moose )
    runPythonTest
}

function makeMoose
{
    make BUILD="$1"
}

function cleanCode 
{
    if [ $# -eq 1 ]; then
        cd "$1" && make clean
    else
        make clean
    fi
}

if [ "$1" = "c" ]; then
    echo "Cleaning $2 and building moose again"
    cleanCode "$2"
    makeMoose developer
    runTest 
elif [ "$1" = "t" ]; then
    echo "Running tests"
    runTest
    exit
else
    echo "Running python test only"
    runPythonTest
fi

