# Gnuplot script
# First run 'moose rall.3.g' to generate *.plot files

set title 'Rallpack 3 (Axon with HH Channels)'
set xlabel 'Step # [dt = 50e-6 s]'
set ylabel 'Vm (V)'
set xrange [0:5000]

# Flash plot for 5 seconds
plot \
	'sim_axon.0' with line title 'Compt 1 (MOOSE)', \
	'ref_axon.0.genesis' using 2 with line title 'Compt 1 (GENESIS)', \
	'sim_axon.x' with line title 'Compt 1000 (MOOSE)', \
	'ref_axon.x.genesis' using 2 with line title 'Compt 1000 (GENESIS)'

pause 5

# Write images to disk
set term png
set output 'axon.0.png'
plot \
	'sim_axon.0' with line title 'Compt 1 (MOOSE)', \
	'ref_axon.0.genesis' using 2 with line title 'Compt 1 (GENESIS)'

set output 'axon.x.png'
plot \
	'sim_axon.x' with line title 'Compt 1000 (MOOSE)', \
	'ref_axon.x.genesis' using 2 with line title 'Compt 1000 (GENESIS)'

set output
set term x11
