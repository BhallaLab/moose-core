#!/bin/bash
echo "Removing any accidentally created cmake files."
rm -rf ../CMakeFiles/ ../CMakeCache.txt
cmake ../
make && make check_python && make check_moose
