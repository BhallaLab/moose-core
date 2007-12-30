#!/bin/tcsh

set MOOSE = ../../moose

# nearDiff is a function to see if two data files are within 
# epsilon of each other. It assumes the files are in xplot format
# If it succeeds, it prints out a period without a newline.
# If it fails, it prints out the first argument and indicates where it
# failed
set NEARDIFF = neardiff

/bin/rm -f test.plot
$MOOSE moose_squid.g > /dev/null
$NEARDIFF moose_squid.plot test.plot 1.0e-5

/bin/rm -f test.plot
$MOOSE moose_kholodenko.g > /dev/null
$NEARDIFF moose_kholodenko.plot test.plot 1.0e-5

/bin/rm -f test.plot
$MOOSE moose_readcell.g > /dev/null
$NEARDIFF moose_readcell.plot test.plot 5.0e-3
