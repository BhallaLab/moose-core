# Gnuplot script
# First run 'genesis BadChannels.g' and 'moose BadChannels.g' to generate *.plot files

set datafile comments '/#'
set xlabel 'Step #'
set ylabel 'Gk (1/ohm)'

# 
# Errant channels:
# 
set title 'Channel: Kca_mit_usb'
plot \
		'Kca_mit_usb.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'Kca_mit_usb.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'Kca_mit_usb.png' ; replot ; set output ; set term x11
print "Plot image written to Kca_mit_usb.png.\n"

set title 'Channel: Ca_bsg_yka'
plot \
		'Ca_bsg_yka.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'Ca_bsg_yka.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'Ca_bsg_yka.png' ; replot ; set output ; set term x11
print "Plot image written to Ca_bsg_yka.png.\n"

set title 'Channel: Ca_hip_traub'
plot \
		'Ca_hip_traub.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'Ca_hip_traub.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'Ca_hip_traub.png' ; replot ; set output ; set term x11
print "Plot image written to Ca_hip_traub.png.\n"

set title 'Channel: Kca_hip_traub'
plot \
		'Kca_hip_traub.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'Kca_hip_traub.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'Kca_hip_traub.png' ; replot ; set output ; set term x11
print "Plot image written to Kca_hip_traub.png.\n"

set title 'Channel: Kc_hip_traub91'
plot \
		'Kc_hip_traub91.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'Kc_hip_traub91.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'Kc_hip_traub91.png' ; replot ; set output ; set term x11
print "Plot image written to Kc_hip_traub91.png.\n"

set title 'Channel: K_hip_traub'
plot \
		'K_hip_traub.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'K_hip_traub.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'K_hip_traub.png' ; replot ; set output ; set term x11
print "Plot image written to K_hip_traub.png.\n"


# 
# Working channels:
# 
set title 'Channel: Ca_hip_traub91'
plot \
		'Ca_hip_traub91.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'Ca_hip_traub91.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'Ca_hip_traub91.png' ; replot ; set output ; set term x11
print "Plot image written to Ca_hip_traub91.png.\n"

set title 'Channel: K2_mit_usb'
plot \
		'K2_mit_usb.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'K2_mit_usb.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'K2_mit_usb.png' ; replot ; set output ; set term x11
print "Plot image written to K2_mit_usb.png.\n"

set title 'Channel: KA_bsg_yka'
plot \
		'KA_bsg_yka.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'KA_bsg_yka.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'KA_bsg_yka.png' ; replot ; set output ; set term x11
print "Plot image written to KA_bsg_yka.png.\n"

set title 'Channel: Ka_hip_traub91'
plot \
		'Ka_hip_traub91.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'Ka_hip_traub91.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'Ka_hip_traub91.png' ; replot ; set output ; set term x11
print "Plot image written to Ka_hip_traub91.png.\n"

set title 'Channel: Kahp_hip_traub91'
plot \
		'Kahp_hip_traub91.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'Kahp_hip_traub91.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'Kahp_hip_traub91.png' ; replot ; set output ; set term x11
print "Plot image written to Kahp_hip_traub91.png.\n"

set title 'Channel: K_bsg_yka'
plot \
		'K_bsg_yka.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'K_bsg_yka.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'K_bsg_yka.png' ; replot ; set output ; set term x11
print "Plot image written to K_bsg_yka.png.\n"

set title 'Channel: Kdr_hip_traub91'
plot \
		'Kdr_hip_traub91.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'Kdr_hip_traub91.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'Kdr_hip_traub91.png' ; replot ; set output ; set term x11
print "Plot image written to Kdr_hip_traub91.png.\n"

set title 'Channel: K_hh_tchan'
plot \
		'K_hh_tchan.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'K_hh_tchan.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'K_hh_tchan.png' ; replot ; set output ; set term x11
print "Plot image written to K_hh_tchan.png.\n"

set title 'Channel: KM_bsg_yka'
plot \
		'KM_bsg_yka.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'KM_bsg_yka.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'KM_bsg_yka.png' ; replot ; set output ; set term x11
print "Plot image written to KM_bsg_yka.png.\n"

set title 'Channel: K_mit_usb'
plot \
		'K_mit_usb.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'K_mit_usb.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'K_mit_usb.png' ; replot ; set output ; set term x11
print "Plot image written to K_mit_usb.png.\n"

set title 'Channel: LCa3_mit_usb'
plot \
		'LCa3_mit_usb.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'LCa3_mit_usb.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'LCa3_mit_usb.png' ; replot ; set output ; set term x11
print "Plot image written to LCa3_mit_usb.png.\n"

set title 'Channel: Na_bsg_yka'
plot \
		'Na_bsg_yka.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'Na_bsg_yka.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'Na_bsg_yka.png' ; replot ; set output ; set term x11
print "Plot image written to Na_bsg_yka.png.\n"

set title 'Channel: Na_hh_tchan'
plot \
		'Na_hh_tchan.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'Na_hh_tchan.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'Na_hh_tchan.png' ; replot ; set output ; set term x11
print "Plot image written to Na_hh_tchan.png.\n"

set title 'Channel: Na_hip_traub91'
plot \
		'Na_hip_traub91.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'Na_hip_traub91.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'Na_hip_traub91.png' ; replot ; set output ; set term x11
print "Plot image written to Na_hip_traub91.png.\n"

set title 'Channel: Na_mit_usb'
plot \
		'Na_mit_usb.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'Na_mit_usb.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'Na_mit_usb.png' ; replot ; set output ; set term x11
print "Plot image written to Na_mit_usb.png.\n"

set title 'Channel: Na_rat_smsnn'
plot \
		'Na_rat_smsnn.genesis.plot' every 3 with linespoints title 'GENESIS', \
		'Na_rat_smsnn.moose.plot' every 3 with linespoints title 'MOOSE'
pause mouse key "Any key to continue.\n"
set term png ; set output 'Na_rat_smsnn.png' ; replot ; set output ; set term x11
print "Plot image written to Na_rat_smsnn.png.\n"
