#!/bin/bash

a=`python -c 'import moose; print([moose.rand() for x in range(10)])'`
b=`python -c 'import moose; print([moose.rand() for x in range(10)])'`

if [[ "$a" == "$b" ]]; then
    echo "Test failed. Expecting not equal output. Got"
    printf "$a \n\t and,\n $b\n"
    exit 1;
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
    exit 1;
fi

exit 0;
