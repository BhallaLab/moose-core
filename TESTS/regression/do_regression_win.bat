@echo off

REM Give absolute location of MOOSE executable here:
set MOOSE=..\..\moose_vcpp2005\moose\Release\moose

GOTO EndComment
nearDiff is a function to see if two data files are within epsilon of each other. It assumes the files are in xplot format.If it succeeds, it prints out a period without a newline.If it fails, it prints out the first argument and indicates where it failed.
:EndComment

set NEARDIFF=.\neardiff

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_squid.g > NUL
ECHO squid
%NEARDIFF% moose_squid.plot test.plot 1.0e-5


ERASE test.plot
%MOOSE% moose_kholodenko.g > NUL
ECHO  kinetics
%NEARDIFF% moose_kholodenko.plot test.plot 1.0e-5


ERASE test.plot
%MOOSE% moose_readcell_global_parms.g >NUL 
ECHO readcell1
%NEARDIFF% moose_readcell_global_parms.plot test.plot 1.0e-5 -fractional

ERASE test.plot
%MOOSE% moose_single_compt.g > NUL
ECHO single_compt
%NEARDIFF% moose_single_compt.plot test.plot 1.0e-3

ERASE test.plot
%MOOSE% moose_readcell.g > NUL
ECHO readcell2
%NEARDIFF% moose_readcell.plot test.plot 5.0e-3

ERASE test.plot
%MOOSE% moose_synchan.g > NUL
ECHO "synchan"
%NEARDIFF% moose_synchan.plot test.plot 5.0e-11

GOTO EndComment1
There is a small numerical divergence between moose and genesis on the upswing of the synchan. This is because of how they handle the timing event arrival, so the MOOSE version is one dt later.
:EndComment1

ERASE test.plot
ECHO "channels"
%MOOSE% moose_channels.g > NUL

GOTO EndComment2
These are all the channels.
Bad channels are
Ca_bsg_yka, Ca_hip_traub, Kca_hip_traub, Kc_hip_traub91, K_hip_traub. 
K_bsg_yka is a little bit off.
foreach i ( Ca_bsg_yka Ca_hip_traub91 Ca_hip_traub K2_mit_usb KA_bsg_yka Ka_hip_traub91 Kahp_hip_traub91 K_bsg_yka Kca_hip_traub Kc_hip_traub91 Kdr_hip_traub91 K_hh_tchan K_hip_traub KM_bsg_yka K_mit_usb LCa3_mit_usb Na_bsg_yka Na_hh_tchan Na_hip_traub91 Na_mit_usb Na_rat_smsnn )
:EndComment2

FOR %%i IN( Ca_hip_traub91 K2_mit_usb KA_bsg_yka Ka_hip_traub91 Kahp_hip_traub91 K_bsg_yka Kdr_hip_traub91 K_hh_tchan KM_bsg_yka K_mit_usb LCa3_mit_usb Na_bsg_yka Na_hh_tchan Na_hip_traub91 Na_mit_usb Na_rat_smsnn ) 	
ECHO %i%
%NEARDIFF% channelplots/moose_%i%.plot test_%i%.plot 5.0e-2 -f
DO

ERASE test.plot
%MOOSE% moose_synapse_solve.g > NUL
ECHO "solver|readcell|synchan"
%NEARDIFF% moose_synapse_solve.plot test.plot 1.0e-3


ERASE test.plot
%MOOSE% moose_network.g > NUL
ECHO "network"
%NEARDIFF% moose_network.plot test.plot 1.0e-11

ERASE test.plot
%MOOSE% moose_file2tab2file.g > NUL
ECHO "file2tab and tab2file"
%NEARDIFF% moose_file2tab.plot test.plot 1.0e-6

ERASE test.plot
ERASE test_*.plot

