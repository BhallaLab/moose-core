# Gnuplot script
# First run 'moose Rallpack1.g' to generate *.plot files

set datafile comments '/#'
set title 'Rallpack 1 (Linear passive cable)'
set xlabel 'Step # [dt = 50e-6 s]'
set ylabel 'Vm (V)'

plot \
	'cable-0.moose.plot' with line title 'Compt 1 (MOOSE)', \
	'cable-0.genesis.plot' with line title 'Compt 1 (GENESIS)', \
	'cable-0.analytical.plot' using 2 with line title 'Compt 1 (Analytical)', \
	'cable-x.moose.plot' with line title 'Compt 1000 (MOOSE)', \
	'cable-x.genesis.plot' with line title 'Compt 1000 (GENESIS)', \
	'cable-x.analytical.plot' using 2 with line title 'Compt 1000 (Analytical)'


pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'cable.png'
replot
set output
set term x11

print "Plot image written to cable.png.\n"
