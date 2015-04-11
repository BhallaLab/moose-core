#!/bin/bash
nrnivmodl
echo "This model runs for 2 second"
time nrniv -nogui ./fig2A.hoc
