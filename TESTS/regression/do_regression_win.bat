@echo off

set MOOSE=..\..\moose_vcpp2005\Debug\moose

REM nearDiff is a function to see if two data files are within epsilon of each other. It assumes the files are in xplot format.If it succeeds, it prints out a period without a newline.If it fails, it prints out the first argument and indicates where it failed.

set NEARDIFF=.\neardiff

IF EXIST regression.out  ERASE regression.out

IF EXIST test.plot ERASE test.plot
%MOOSE%  -p NeuroML_Reader moose_NeuroML_reader.g >> regression.out 2>&1
%NEARDIFF% moose_NeuroMLReader.plot test.plot 5.0e-3
ECHO NeuroML_Read

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_sbml_reader.g >> regression.out 2>&1
%NEARDIFF% acc88_copasi.plot test.plot 3.0e-2 -f
ECHO sbml_Read

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_sbml_read_write.g >> regression.out 2>&1
%NEARDIFF% moose.plot test.plot 1.0e-16 
ECHO sbml_Read_Write

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_kholodenko.g >> regression.out 2>&1
%NEARDIFF% moose_kholodenko.plot test.plot 1.0e-5
ECHO  kinetics

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_squid.g >> regression.out 2>&1
%NEARDIFF% moose_squid.plot test.plot 1.0e-5
ECHO squid

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_readcell_global_parms.g >> regression.out 2>&1
%NEARDIFF% moose_readcell_global_parms.plot test.plot 1.0e-5 -fractional
ECHO readcell1

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_readcell.g >> regression.out 2>&1
%NEARDIFF% moose_readcell.plot test.plot 5.0e-3
ECHO readcell2

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_synchan.g >> regression.out 2>&1
%NEARDIFF% moose_synchan.plot test.plot 5.0e-11
ECHO synchan
REM There is a small numerical divergence between moose and genesis on the upswing of the synchan. This is because of how they handle the timing event arrival, so the MOOSE version is one dt later.

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_channels.g >> regression.out 2>&1
REM # Bad channels are
REM # Ca_bsg_yka, Ca_hip_traub, Kca_hip_traub, Kc_hip_traub91, K_hip_traub. 
REM # K_bsg_yka is a little bit off.
FOR %%i IN ( Ca_hip_traub91 K2_mit_usb KA_bsg_yka Ka_hip_traub91 Kahp_hip_traub91 K_bsg_yka Kdr_hip_traub91 K_hh_tchan KM_bsg_yka K_mit_usb LCa3_mit_usb Na_bsg_yka Na_hh_tchan Na_hip_traub91 Na_mit_usb Na_rat_smsnn ) DO %NEARDIFF% channelplots\moose_%%i.plot test_%%i.plot 5.0e-2 -f

ECHO channels
GOTO C2

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_network.g >> regression.out 2>&1
%NEARDIFF% moose_network.plot test.plot 1.0e-11
ECHO network
:C2

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_axon.g >> regression.out 2>&1
%NEARDIFF% moose_axon.plot test.plot 5.0e-3
ECHO axon

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_mitral.g >> regression.out 2>&1
%NEARDIFF% moose_mitral.plot test.plot 5.0e-3
ECHO mitral	
	
IF EXIST test.plot ERASE test.plot

%MOOSE% moose_rallpack1.g >> regression.out 2>&1
%NEARDIFF% moose_rallpack1.plot test.plot 5.0e-5
ECHO rallpack1	
	
IF EXIST test.plot ERASE test.plot

%MOOSE% moose_rallpack2.g >> regression.out 2>&1
%NEARDIFF% moose_rallpack2.plot test.plot 5.0e-6
ECHO rallpack2
	
IF EXIST test.plot ERASE test.plot

%MOOSE% moose_rallpack3.g >> regression.out 2>&1
%NEARDIFF% moose_rallpack3.plot test.plot 5.0e-2
ECHO rallpack3
	
IF EXIST test.plot ERASE test.plot

%MOOSE% moose_synapse.g >> regression.out 2>&1
%NEARDIFF% moose_synapse.plot test.plot 5.0e-7
ECHO synapse
	
IF EXIST test.plot ERASE test.plot

%MOOSE% moose_traub91.g >> regression.out 2>&1
%NEARDIFF% moose_traub91.plot test.plot 5.0e-2
ECHO traub91

IF EXIST test.plot ERASE test.plot

%MOOSE% moose_file2tab2file.g >> regression.out 2>&1
%NEARDIFF% moose_file2tab.plot test.plot 1.0e-6
ECHO file2tab and tab2file
