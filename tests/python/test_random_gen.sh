#!/bin/bash

FAILED=0
a=`python -c 'import moose; moose.seed(1); print([moose.rand() for x in range(10)])'`
b=`python -c 'import moose; moose.seed(2); print([moose.rand() for x in range(10)])'`

if [[ "$a" == "$b" ]]; then
    echo "Test 1 failed. Expecting not equal output. Got"
    printf "$a \n\t and,\n $b\n"
    FAILED=1
else
    echo "Test 1 passed"
fi

c=`python -c 'import moose; moose.seed(10); print([moose.rand() for x in \
    range(10)])'`
d=`python -c 'import moose; moose.seed(10); print([moose.rand() for x in \
    range(10)])'`

if [[ "$c" == "$d" ]]; then
    echo "Test 2 passed"
else
    echo "Test failed. Expecting equal output. Got"
    printf "$c \n\t and,\n$d\n"
    FAILED=1
fi

if [ $FAILED -eq 1 ]; then 
    exit 1
else
    exit 0;
fi
