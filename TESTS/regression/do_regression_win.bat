@echo off

set MOOSE=..\..\moose_vcpp2005\Release\moose

REM nearDiff is a function to see if two data files are within epsilon of each other. It assumes the files are in xplot format.If it succeeds, it prints out a period without a newline.If it fails, it prints out the first argument and indicates where it failed.

set NEARDIFF=.\neardiff

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_squid.g > NUL
%NEARDIFF% moose_squid.plot test.plot 1.0e-5
ECHO squid

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_sbml_reader.g > NUL
%NEARDIFF% acc88_copasi.plot test.plot 3.0e-2 -f
ECHO sbml_Read

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_sbml_read_write.g > NUL
%NEARDIFF% moose.plot test.plot 1.0e-16 
ECHO sbml_Read_Write


IF EXIST test.plot ERASE test.plot

%MOOSE% moose_kholodenko.g > NUL
%NEARDIFF% moose_kholodenko.plot test.plot 1.0e-5
ECHO  kinetics

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_readcell_global_parms.g >NUL 
%NEARDIFF% moose_readcell_global_parms.plot test.plot 1.0e-5 -fractional
ECHO readcell1

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_single_compt.g > NUL
%NEARDIFF% moose_single_compt.plot test.plot 5.0e-3
ECHO single_compt

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_readcell.g > NUL
%NEARDIFF% moose_readcell.plot test.plot 5.0e-3
ECHO readcell2

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_synchan.g > NUL
%NEARDIFF% moose_synchan.plot test.plot 5.0e-11
ECHO "synchan"
REM There is a small numerical divergence between moose and genesis on the upswing of the synchan. This is because of how they handle the timing event arrival, so the MOOSE version is one dt later.

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_channels.g > NUL
ECHO "channels"
REM # Bad channels are
REM # Ca_bsg_yka, Ca_hip_traub, Kca_hip_traub, Kc_hip_traub91, K_hip_traub. 
REM # K_bsg_yka is a little bit off.
FOR %%i IN ( Ca_hip_traub91 K2_mit_usb KA_bsg_yka Ka_hip_traub91 Kahp_hip_traub91 K_bsg_yka Kdr_hip_traub91 K_hh_tchan KM_bsg_yka K_mit_usb LCa3_mit_usb Na_bsg_yka Na_hh_tchan Na_hip_traub91 Na_mit_usb Na_rat_smsnn ) DO %NEARDIFF% channelplots\moose_%%i.plot test_%%i.plot 5.0e-2 -f

GOTO C2
IF EXIST test.plot ERASE test.plot

%MOOSE% moose_synapse_solve.g > NUL
%NEARDIFF% moose_synapse_solve.plot test.plot 1.0e-3
ECHO "solver|readcell|synchan"

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_network.g > NUL
%NEARDIFF% moose_network.plot test.plot 1.0e-11
ECHO "network"
:C2

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_file2tab2file.g > NUL
%NEARDIFF% moose_file2tab.plot test.plot 1.0e-6
ECHO "file2tab and tab2file"

IF EXIST test.plot ERASE test.plot
ERASE test_*.plot
