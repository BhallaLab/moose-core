#!/bin/bash
set -e 
# Definitions of colors in bash
RESTORE='\033[0m'

RED='\033[00;31m'
GREEN='\033[00;32m'
YELLOW='\033[00;33m'
BLUE='\033[00;34m'
PURPLE='\033[00;35m'
CYAN='\033[00;36m'
LIGHTGRAY='\033[00;37m'

LRED='\033[01;31m'
LGREEN='\033[01;32m'
LYELLOW='\033[01;33m'
LBLUE='\033[01;34m'
LPURPLE='\033[01;35m'
LCYAN='\033[01;36m'
WHITE='\033[01;37m'

function colorPrint
{
    case $1 in 
        "WARN")
            echo -e "[WARN] ${LRED} $2 ${RESTORE} $3"
            ;;
        "INFO")
            echo -e "[INFO] ${LGREEN} $2 ${RESTORE} $3"
            ;;
        "ERROR")
            echo -e "[ERROR] ${RED} $2 ${RESTORE} $3"
            ;;
        "DEBUG")
            echo -e "[DEBUG] ${YELLOW} $2 ${RESTORE} $3"
            ;;
        "STEP")
            echo -e "[STEP] ${BLUE} $2 ${RESTORE} $3"
            ;;
        "TODO")
            echo -e "[TODO] ${CYAN} $2 ${RESTORE} $3"
            ;;
        *)
            echo -e "[$1] $2 $3"
            ;;
    esac
}
colorPrint "INFO" "Beware, O unsuspecting developer!" 
colorPrint "INFO" "This script builds moose using cmake. If you want to use standard "
colorPrint "INFO" "version, run make"
colorPrint "INFO" "You may like to customize some varibales in Makefile."
colorPrint "INFO" ""

BUILD_TYPE=Debug
if [ "$1" ]; then
    colorPrint "STEP" "Building RELEASE version"
    export CXX_FLAGS=-O3
    BUILD_TYPE=RELEASE 
fi
( cd ./buildMooseUsingCmake && \
    ./build_me.sh -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DQUIET_MODE 
)
