#!/bin/bash
set -e # Stop at first error

echo "++ This scripts run rallpacks. The python script should be used by end-user."
echo "++ We  are using this sccript for testing and development purpsose"

PYC=python2.7

function runRallPack1 
{
    echo "Running rallpac1"
    $PYC ./cable.py 
    if [ -f cable.spice ]; then
        ngspice -b ./cable.spice
    fi
}

function runRallPack2 
{
    if [ ! $1 ]; then
        echo "[ERROR] Missing depth of cable. Please provide one."
        return
    fi
    echo "[STEP] Running rallpack2 for cable depth $1"
    $PYC ./tree_cable.py $1
    echo "Turning graphviz into eps"
    dot -Teps -Nshape=point ./figures/binary_tree.dot > ./figures/binary_tree.eps
}

if [ $1 -eq 1 ]; then 
    runRallPack1
elif [ $1 -eq 2 ]; then
    runRallPack2 $2
else
    runRallPack1
    runRallPack2 $2
fi
