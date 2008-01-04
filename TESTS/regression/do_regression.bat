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

/bin/rm -f test.plot
$MOOSE moose_channels.g > /dev/null

######################################################################
# These are all the channels.
# Bad channels are 
# Ca_bsg_yka, Ca_hip_traub, Kca_hip_traub, Kc_hip_traub91, K_hip_traub
# K_bsg_yka is a little bit off.
#
#foreach i ( Ca_bsg_yka Ca_hip_traub91 Ca_hip_traub K2_mit_usb KA_bsg_yka Ka_hip_traub91 Kahp_hip_traub91 K_bsg_yka Kca_hip_traub Kc_hip_traub91 Kdr_hip_traub91 K_hh_tchan K_hip_traub KM_bsg_yka K_mit_usb LCa3_mit_usb Na_bsg_yka Na_hh_tchan Na_hip_traub91 Na_mit_usb Na_rat_smsnn )
######################################################################

foreach i ( Ca_hip_traub91 K2_mit_usb KA_bsg_yka Ka_hip_traub91 Kahp_hip_traub91 K_bsg_yka Kdr_hip_traub91 K_hh_tchan KM_bsg_yka K_mit_usb LCa3_mit_usb Na_bsg_yka Na_hh_tchan Na_hip_traub91 Na_mit_usb Na_rat_smsnn )
	$NEARDIFF moose_$i.plot test_$i.plot 5.0e-2 -f
end

/bin/rm -f test_*.plot
