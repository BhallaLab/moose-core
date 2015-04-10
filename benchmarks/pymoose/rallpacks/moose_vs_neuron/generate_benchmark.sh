#!/bin/bash
# This script generates benchamark.
set -x
set -e
if [[ $# -lt 2 ]]; then
    echo "USAGE: $0 dir_name"
    exit
fi
dir="$1"
( 
echo "Working in $dir"
cd $dir
for i in `seq 500 500 35000`
do
    for j in `seq 1 1 3` 
    do
        python moose_sim.py --run_time 0.25 --ncomp $i
        python neuron_sim.py --run_time 0.25 --ncomp $i
    done
done
)
