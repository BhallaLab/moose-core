This directory has regression tests for MOOSE. These are tests which 
exercise some aspect of MOOSE function by running a specific model.
The idea is that all these tests should be run automatically with a single
command.

To run the regression tests, type 'source do_regression.bat'

To add a new test to the ones here, you will need to do the following:

1. Set up a model to test. Ideally this should be a single script file.
2. Tweak the script of the model so that it automatically generates an output 
	file called 'test.plot', and then quits.
3. Generate a plot of your reference output. This should have the same name
	as your script file, but with the suffix .plot instead of the .g
4. Append three lines to the script 'do_regression.bat' in the format you see
	for earlier scripts.


=============================================================================

The current regression tests are:
1. A version of the squid model. This only runs with MOOSE
2. An oscillatory kinetic model, due to Boris Kholodenko
3. A simplified cell model to check how readcell handles globals
4. A full cell model and readcell, using the exponential Euler (EE) method
5. A series of tests for a large number of ion channels, using EE.
	This family of ion channels should cover many kinds that GENESIS users
	would like to implement.
	There is also a list of the channels that do NOT work.
6. A 2 cell model with a single synapse in between.
	Uses hsolve
	Uses readcell
7. Simple network of single compartment neurons. Uses createmap and
   planarconnect. Does not use solver.
8. Test of tab2file and file2tab commands.

=============================================================================
