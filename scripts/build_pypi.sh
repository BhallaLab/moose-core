#!/bin/bash
set -e
# This creates a package for pip. For testing purpose
( 
    cd ..  
    svn export --force . scripts/moose-3.0 
)
(
    cd moose-3.0
    echo "Creating new archive"
    if [ -f dist/moose-3.0.tar.gz ]; then
        rm -f dist/*.tar.gz
    fi
    python setup.py sdist -vv
    echo "Created new archive"
    cd dist && pip install -vvvv *.tar.gz --user
)
