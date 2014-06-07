#!/bin/bash
set -e 
source scripts/color.sh
colorPrint "||" "Beware, O unsuspecting developer!" 
colorPrint "||" "This script builds moose using cmake. If you want to use standard "
colorPrint "||" "version, run make"
colorPrint "||" "You may like to customize some varibales in Makefile."

BUILD_TYPE=Debug
if [ $# -gt 0 ]; then
    colorPrint "STEP" "Building RELEASE version"
    export CXX_FLAGS=-O3
    BUILD_TYPE=RELEASE 
fi
(
    cd ./buildMooseUsingCmake && ./build_me.sh
)
