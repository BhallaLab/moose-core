#!/bin/bash
set -e 
source ../scripts/color.sh

BUILD_TYPE=debug
if [ $# -gt 0 ]; then
    colorPrint "INPUT" "Building for distribution" 
    BUILD_TYPE=distribution 
fi
rm -rf ../CMakeFiles/ 
rm -f ../CMakeCache.txt
rm -f CMakeCache.txt
cmake ..
make VERBOSE=0
