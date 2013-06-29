#!/bin/bash

clear

rm results0.dat -f
rm results1.dat -f

rm testDir -rf 
mkdir testDir

neurons=1000
seed=6
type=n1l

#
#echo "1) Building non-MPI version"
#rm *.o HinesGpu -f
#./buildLinux.sh > testDir/buildLinux.dat 2>&1

#
echo "- Running version with 1 thread [C|G|H]"
./testHines H $type $neurons 1 $seed > testDir/H1.dat 2>&1
cat testDir/H1.dat | grep meanGenSpikes

echo "- Running version with 2 GPUs [C|G|H]"
./testHines H $type $neurons 1 $seed > testDir/H2.dat 2>&1
cat testDir/H2.dat | grep meanGenSpikes

#1) Building non-MPI version
#- Running version with 1 thread [C|G|H]
#meanGenSpikes[T|P|I|B]=[9.84|11.05|9.26|9.20]
#meanGenSpikes[T|P|I|B]=[9.84|11.05|9.26|9.20]
#meanGenSpikes[T|P|I|B]=[9.84|11.05|9.26|9.20]
#- Running version with 2 GPUs [C|G|H]
#meanGenSpikes[T|P|I|B]=[9.84|11.05|9.26|9.20]
#meanGenSpikes[T|P|I|B]=[9.84|11.05|9.26|9.20]
#meanGenSpikes[T|P|I|B]=[9.84|11.05|9.26|9.20]
