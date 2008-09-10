# Gnuplot script
# First run 'moose rall.2.g' to generate *.plot files

set datafile comments '/#'
set title 'Rallpack 2 (Branched passive cable)'
set xlabel 'Step # [dt = 50e-6 s]'
set ylabel 'Vm (V)'

# Flash plot for 5 seconds
plot \
	'sim_branch.0' with line title 'Compt 1 (MOOSE)', \
	'ref_branch.0' using 2 with line title 'Compt 1 (Analytical)', \
	'sim_branch.x' with line title 'Compt 1023 (MOOSE)', \
	'ref_branch.x' using 2 with line title 'Compt 1023 (Analytical)'

pause 5

# Write images to disk
set term png
set output 'branch.0.png'
plot \
	'sim_branch.0' with line title 'Compt 1 (MOOSE)', \
	'ref_branch.0' using 2 with line title 'Compt 1 (Analytical)'

set output 'branch.x.png'
plot \
	'sim_branch.x' with line title 'Compt 1023 (MOOSE)', \
	'ref_branch.x' using 2 with line title 'Compt 1023 (Analytical)'

set output
set term x11
