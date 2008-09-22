# Gnuplot script
# First run 'moose Channels.g' to generate *.plot files

set datafile comments '/#'
set xlabel 'Step # [dt = 50e-6 s]'
set ylabel 'Gk (1/ohm)'

set title 'Channel: Ca_bsg_yka'
plot \
		'reference_plots/Ca_bsg_yka.genesis.plot' with line, \
		'Ca_bsg_yka.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: Ca_hip_traub91'
plot \
		'reference_plots/Ca_hip_traub91.genesis.plot' with line, \
		'Ca_hip_traub91.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: Ca_hip_traub'
plot \
		'reference_plots/Ca_hip_traub.genesis.plot' with line, \
		'Ca_hip_traub.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: K2_mit_usb'
plot \
		'reference_plots/K2_mit_usb.genesis.plot' with line, \
		'K2_mit_usb.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: KA_bsg_yka'
plot \
		'reference_plots/KA_bsg_yka.genesis.plot' with line, \
		'KA_bsg_yka.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: Ka_hip_traub91'
plot \
		'reference_plots/Ka_hip_traub91.genesis.plot' with line, \
		'Ka_hip_traub91.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: Kahp_hip_traub91'
plot \
		'reference_plots/Kahp_hip_traub91.genesis.plot' with line, \
		'Kahp_hip_traub91.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: K_bsg_yka'
plot \
		'reference_plots/K_bsg_yka.genesis.plot' with line, \
		'K_bsg_yka.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: Kca_hip_traub'
plot \
		'reference_plots/Kca_hip_traub.genesis.plot' with line, \
		'Kca_hip_traub.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: Kc_hip_traub91'
plot \
		'reference_plots/Kc_hip_traub91.genesis.plot' with line, \
		'Kc_hip_traub91.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: Kdr_hip_traub91'
plot \
		'reference_plots/Kdr_hip_traub91.genesis.plot' with line, \
		'Kdr_hip_traub91.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: K_hh_tchan'
plot \
		'reference_plots/K_hh_tchan.genesis.plot' with line, \
		'K_hh_tchan.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: K_hip_traub'
plot \
		'reference_plots/K_hip_traub.genesis.plot' with line, \
		'K_hip_traub.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: Ca_bsg_yka'
plot \
		'reference_plots/Ca_bsg_yka.genesis.plot' with line, \
		'Ca_bsg_yka.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: KM_bsg_yka'
plot \
		'reference_plots/KM_bsg_yka.genesis.plot' with line, \
		'KM_bsg_yka.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: K_mit_usb'
plot \
		'reference_plots/K_mit_usb.genesis.plot' with line, \
		'K_mit_usb.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: LCa3_mit_usb'
plot \
		'reference_plots/LCa3_mit_usb.genesis.plot' with line, \
		'LCa3_mit_usb.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: Ca_bsg_yka'
plot \
		'reference_plots/Ca_bsg_yka.genesis.plot' with line, \
		'Ca_bsg_yka.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: Na_bsg_yka'
plot \
		'reference_plots/Na_bsg_yka.genesis.plot' with line, \
		'Na_bsg_yka.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: Na_hh_tchan'
plot \
		'reference_plots/Na_hh_tchan.genesis.plot' with line, \
		'Na_hh_tchan.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: Na_hip_traub91'
plot \
		'reference_plots/Na_hip_traub91.genesis.plot' with line, \
		'Na_hip_traub91.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: Na_mit_usb'
plot \
		'reference_plots/Na_mit_usb.genesis.plot' with line, \
		'Na_mit_usb.moose.plot' w l
pause mouse key "Any key to continue.\n"

set title 'Channel: Na_rat_smsnn'
plot \
		'reference_plots/Na_rat_smsnn.genesis.plot' with line, \
		'Na_rat_smsnn.moose.plot' w l
pause mouse key "Any key to continue.\n"
