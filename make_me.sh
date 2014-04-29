#!/bin/bash
if [ $# -eq 1 ]; then
    make clean
fi
make BUILD=developer
