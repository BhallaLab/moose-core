#!/bin/bash

##########################
## 'time' arguments
##########################
## FORMAT STRING:
## Time (in seconds):
## %E: Real time
## %U: User time
## %S: Sys time
##
## Memory (in KiB):
## %M: Max. resident set size
## %t: Avg. resident set size
## %K: Avg. total memory use (data+stack+text)
##
## %C: Command and its command-line args

## Write in file
alias TIME='/usr/bin/time -f "%E\t%U\t%S\t%C" --output=../results --append'
## Write to stdout
#~ alias TIME='/usr/bin/time -f "%E\t%U\t%S\t%C"'

## Write memory usage
#~ alias TIME='/usr/bin/time -f "%E\t%U\t%S\t%M\t%t\t%K\t%C"'

##########################
## Choose simulator
##########################
#SIM='genesis'
 SIM='mpirun ../../moose'
##########################
## Benchmarks
##########################

##########################
## Rallpack 1
##########################
#cd rallpack1
#SCRIPT='rall.1.g'
#SIMLENGTH=0.5
#PLOT=0 # Do not generate plots

##
## Vary number of compartments
##
#NCELL=1

## First measure time for the setup only (Runtime = 0.0)
#for SIZE in  1 2 5 10 20 50 100 200 500 1000 2000 5000 10000;
#do
#	ARGS=$NCELL' '$SIZE' '0.0' '$PLOT
#	echo $SCRIPT $ARGS
#	TIME $SIM $SCRIPT $ARGS > /dev/null
#done

## Then run the benchmarks
#for SIZE in 1 2 5 10 20 50 100 200 500 1000 2000 5000 10000;
#do
#	ARGS=$NCELL' '$SIZE' '$SIMLENGTH' '$PLOT
#	echo $SCRIPT $ARGS
#	TIME $SIM $SCRIPT $ARGS > /dev/null
#done

##
## Vary number of cells and also size of cells
##

## First measure time for the setup only (Runtime = 0.0)
#for NCELL in 1 2 5 10 20 50 100;
#do
#	for SIZE in 10 100 1000;
#	do
#		ARGS=$NCELL' '$SIZE' '0.0' '$PLOT
#		echo $SCRIPT $ARGS
#		TIME $SIM $SCRIPT $ARGS > /dev/null
#	done
#done

## Then run the benchmarks
#for NCELL in 1 2 5 10 20 50 100;
#do
#	for SIZE in 10 100 1000;
#	do
#		ARGS=$NCELL' '$SIZE' '$SIMLENGTH' '$PLOT
#		echo $SCRIPT $ARGS
#		TIME $SIM $SCRIPT $ARGS > /dev/null
#	done
#done
#cd ..

###########################
### Rallpack 2
###########################
#cd rallpack2
#SCRIPT='rall.2.g'
#SIMLENGTH=0.5
#PLOT=0 # Do not generate plots
#
###
### Vary number of compartments
###
#NCELL=1
#
### First measure time for the setup only (Runtime = 0.0)
#for SIZE in 1 2 3 4 5 6 7 8 9 10 11 12 13;
#do
#	ARGS=$NCELL' '$SIZE' '0.0' '$PLOT
#	echo $SCRIPT $ARGS
#	TIME $SIM $SCRIPT $ARGS > /dev/null
#done

## Then run the benchmarks
#for SIZE in 1 2 3 4 5 6 7 8 9 10 11 12 13;
#do
#	ARGS=$NCELL' '$SIZE' '$SIMLENGTH' '$PLOT
#	echo $SCRIPT $ARGS
#	TIME $SIM $SCRIPT $ARGS > /dev/null
#done

###
### Vary number of cells and also size of cells
###

### First measure time for the setup only (Runtime = 0.0)
#for NCELL in 1 2 5 10 20 50 100;
#do
#	for SIZE in 1 3 5 7 9;
#	do
#		ARGS=$NCELL' '$SIZE' '0.0' '$PLOT
#		echo $SCRIPT $ARGS
#		TIME $SIM $SCRIPT $ARGS > /dev/null
#	done
#done

### Then run the benchmarks
#for NCELL in 1 2 5 10 20 50 100;
#do
#	for SIZE in 1 3 5 7 9;
#	do
#		ARGS=$NCELL' '$SIZE' '$SIMLENGTH' '$PLOT
#		echo $SCRIPT $ARGS
#		TIME $SIM $SCRIPT $ARGS > /dev/null
#	done
#done
#cd ..
#
###########################
### Traub's CA3 model
###########################
#cd traub91
#SCRIPT='traub91.g'
#SIMLENGTH=0.5
#PLOT=0 # Do not generate plots
#SIZE=1 # No need to specify size for traub91.g. Give any value.

### First measure time for the setup only (Runtime = 0.0)
#for NCELL in 1 2 5 10 20 50 100 200 500 1000;
#do
#	ARGS=$NCELL' '$SIZE' '0.0' '$PLOT
#	echo $SCRIPT $ARGS
#	TIME $SIM $SCRIPT $ARGS > /dev/null
#done

## Then run the benchmarks
#for NCELL in 1 2 5 10 20 50 100 200 500 1000;
#do
#	ARGS=$NCELL' '$SIZE' '$SIMLENGTH' '$PLOT
#	echo $SCRIPT $ARGS
#	TIME $SIM $SCRIPT $ARGS > /dev/null
#done
#cd ..

############################
### Myelin
###########################
cd myelin
SCRIPT='Myelin.g'
SIMLENGTH=0.5
PLOT=0 # Do not generate plots
SIZE=1 # No need to specify size for Myelin.g. Give any value.

### First measure time for the setup only (Runtime = 0.0)
for NCELL in 1 2 5 10 20 50 100;
do
	ARGS=$NCELL' '$SIZE' '0.0' '$PLOT
	echo $SCRIPT $ARGS
	TIME $SIM $SCRIPT $ARGS > /dev/null
done

### Then run the benchmarks
for NCELL in 1 2 5 10 20 50 100;
do
	ARGS=$NCELL' '$SIZE' '$SIMLENGTH' '$PLOT
	echo $SCRIPT $ARGS
	TIME $SIM $SCRIPT $ARGS > /dev/null
done
cd ..
