# Gnuplot script
# First run 'moose rall.1.g' to generate *.plot files

set datafile comments '/#'
set title 'Rallpack 1 (Linear passive cable)'
set xlabel 'Step # [dt = 50e-6 s]'
set ylabel 'Vm (V)'

# Flash plot for 5 seconds
plot \
	'sim_cable.0' with line title 'Compt 1 (MOOSE)', \
	'ref_cable.0' using 2 with line title 'Compt 1 (Analytical)', \
	'sim_cable.x' with line title 'Compt 1000 (MOOSE)', \
	'ref_cable.x' using 2 with line title 'Compt 1000 (Analytical)'

pause 5

# Write images to disk
set term png
set output 'cable.png'
replot
set output
set term x11
