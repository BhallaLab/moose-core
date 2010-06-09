#!/bin/tcsh

set mpi = 0
set sbml = 0
set neuroml = 0
set neuronal = 1
set kinetics = 0
set misc = 0

if ( $mpi ) then
	set MOOSE = 'mpirun -n 1 ../../moose'
else
	set MOOSE = '../../moose'
endif

# nearDiff is a function to see if two data files are within 
# epsilon of each other. It assumes the files are in xplot format
# If it succeeds, it prints out a period without a newline.
# If it fails, it prints out the first argument and indicates where it
# failed
set NEARDIFF = ./neardiff

################################################################################
# Cleanup
################################################################################
# All output from the regression tests is redirected to the file 'regression.out'.
# The symbol >>& instructs the shell to append stdout and stderr to the given file.
/bin/rm -f regression.out >& /dev/null
/bin/rm -f test.plot >& /dev/null
/bin/rm -f test_*.plot >& /dev/null

################################################################################
# NeuroML section
################################################################################
if ( $neuroml ) then
	echo '\n===== NeuroML tests ====='
	echo -n NeuroML_Read
	$MOOSE -p NeuroML_Reader moose_NeuroML_reader.g >>& regression.out
	$NEARDIFF moose_NeuroMLReader.plot test.plot 5.0e-3
	echo
	/bin/rm -f test.plot >& /dev/null
endif


################################################################################
# SBML section
################################################################################
if ( $sbml ) then
	echo '\n===== SBML tests ====='
	echo -n sbml_Read
	$MOOSE moose_sbml_reader.g >>& regression.out
	$NEARDIFF acc88_copasi.plot test.plot 3.0e-2 -f
	echo
	/bin/rm -f test.plot >& /dev/null

	#### v20 model gives 5% error when simulating for 3000 secs and 10% for 15000 secs ####
	#echo -n sbml_Read
	#$MOOSE moose_sbml_reader.g >>& regression.out
	#$NEARDIFF v20_copasi.plot v20_moose.plot 5.0e-2 -f
	#echo
	#/bin/rm -f test.plot >& /dev/null

	echo -n sbml_Read_Write
	$MOOSE moose_sbml_read_write.g >>& regression.out
	$NEARDIFF moose.plot test.plot 1.0e-16 
	echo
	/bin/rm -f test.plot >& /dev/null
endif

################################################################################
# Kinetics section
################################################################################
if ( $kinetics ) then
	echo '\n===== Kinetics tests ====='
	echo -n kinetics
	$MOOSE moose_kholodenko.g >>& regression.out
	$NEARDIFF moose_kholodenko.plot test.plot 1.0e-5
	echo
	/bin/rm -f test.plot >& /dev/null
endif

################################################################################
# Neuronal section
################################################################################
if ( $neuronal ) then
	echo '\n===== Neuronal tests ====='
	echo -n squid
	$MOOSE moose_squid.g >>& regression.out
	$NEARDIFF moose_squid.plot test.plot 5.0e-6
	echo 
	/bin/rm -f test.plot >& /dev/null

	echo -n readcell1
	$MOOSE moose_readcell_global_parms.g >>& regression.out
	$NEARDIFF moose_readcell_global_parms.plot test.plot 1.0e-5 -fractional
	echo
	/bin/rm -f test.plot >& /dev/null

	echo -n readcell2
	$MOOSE moose_readcell.g >>& regression.out
	$NEARDIFF moose_readcell.plot test.plot 5.0e-3
	echo
	/bin/rm -f test.plot >& /dev/null

	echo -n "synchan"
	$MOOSE moose_synchan.g >>& regression.out
	$NEARDIFF moose_synchan.plot test.plot 5.0e-11
	echo
	/bin/rm -f test.plot >& /dev/null
	# There is a small numerical divergence between moose and genesis
	# on the upswing of the synchan. This is because of how they handle the
	# timing event arrival, so the MOOSE version is one dt later.

	echo "channels:"
	# Redirect error messages to regression.out (gives many errors which can be ignored).
	$MOOSE moose_channels.g >>& regression.out

	######################################################################
	# These are all the channels.
	# Bad channels are 
	# Ca_bsg_yka, Ca_hip_traub, Kca_hip_traub, Kc_hip_traub91, K_hip_traub
	# K_bsg_yka is a little bit off.
	#
	#foreach i ( Ca_bsg_yka Ca_hip_traub91 Ca_hip_traub K2_mit_usb KA_bsg_yka Ka_hip_traub91 Kahp_hip_traub91 K_bsg_yka Kca_hip_traub Kc_hip_traub91 Kdr_hip_traub91 K_hh_tchan K_hip_traub KM_bsg_yka K_mit_usb LCa3_mit_usb Na_bsg_yka Na_hh_tchan Na_hip_traub91 Na_mit_usb Na_rat_smsnn )
	######################################################################

	foreach i ( Ca_hip_traub91 K2_mit_usb KA_bsg_yka Ka_hip_traub91 Kahp_hip_traub91 K_bsg_yka Kdr_hip_traub91 K_hh_tchan KM_bsg_yka K_mit_usb LCa3_mit_usb Na_bsg_yka Na_hh_tchan Na_hip_traub91 Na_mit_usb Na_rat_smsnn )
		echo -n '\t'$i
		$NEARDIFF channelplots/moose_$i.plot test_$i.plot 5.0e-2 -f
		echo
	end
	/bin/rm -f test_*.plot >& /dev/null

	echo -n "network"
	$MOOSE moose_network.g >>& regression.out
	$NEARDIFF moose_network.plot test.plot 1.0e-3
	echo
	/bin/rm -f test.plot >& /dev/null

	echo -n "axon"
	$MOOSE moose_axon.g >>& regression.out
	$NEARDIFF moose_axon.plot test.plot 5.0e-3
	echo
	/bin/rm -f test.plot >& /dev/null

	echo -n "mitral"
	$MOOSE moose_mitral.g >>& regression.out
	$NEARDIFF moose_mitral.plot test.plot 5.0e-3
	echo
	/bin/rm -f test.plot >& /dev/null

	echo -n "rallpack1"
	$MOOSE moose_rallpack1.g >>& regression.out
	$NEARDIFF moose_rallpack1.plot test.plot 5.0e-5
	echo
	/bin/rm -f test.plot >& /dev/null

	echo -n "rallpack2"
	$MOOSE moose_rallpack2.g >>& regression.out
	$NEARDIFF moose_rallpack2.plot test.plot 5.0e-6
	echo
	/bin/rm -f test.plot >& /dev/null

	echo -n "rallpack3"
	$MOOSE moose_rallpack3.g >>& regression.out
	$NEARDIFF moose_rallpack3.plot test.plot 5.0e-2
	echo
	/bin/rm -f test.plot >& /dev/null

	echo -n "synapse"
	$MOOSE moose_synapse.g >>& regression.out
	$NEARDIFF moose_synapse.plot test.plot 5.0e-7
	echo
	/bin/rm -f test.plot >& /dev/null

	echo -n "traub91"
	$MOOSE moose_traub91.g >>& regression.out
	$NEARDIFF moose_traub91.plot test.plot 5.0e-2
	echo
	/bin/rm -f test.plot >& /dev/null
endif

################################################################################
# Miscellaneous section
################################################################################
if ( $misc ) then
	echo '\n===== Misc. tests ====='
	echo -n "file2tab and tab2file"
	$MOOSE moose_file2tab2file.g >>& regression.out
	$NEARDIFF moose_file2tab.plot test.plot 1.0e-6
	echo
	/bin/rm -f test.plot >& /dev/null
endif
