#!/bin/bash
echo "Removing any accidentally created cmake files."
rm -rf ../CMakeFiles/ ../CMakeCache.txt
cmake -DVERBOSITY=2 ../
make && make check_moose
