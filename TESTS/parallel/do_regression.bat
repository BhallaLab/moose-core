#!/bin/tcsh

#set NUM_NODES = $1

set MOOSE = ../../moose

# nearDiff is a function to see if two data files are within 
# epsilon of each other. It assumes the files are in xplot format
# If it succeeds, it prints out a period without a newline.
# If it fails, it prints out the first argument and indicates where it
# failed
set NEARDIFF = ../regression/neardiff

foreach NUM_NODES ( 1 2 4 8 )
	# First test checks the most basic element manipulation operations
	echo -n element_manipulation
	mpirun -np $NUM_NODES $MOOSE element_manipulation.g $NUM_NODES

	echo -n showField
	mpirun -np $NUM_NODES $MOOSE showField.g $NUM_NODES
end
