# Gnuplot script
# First run 'moose rall.1.g' to generate *.plot files

set title 'Myelinated Axon'
set xlabel 'Step # [dt = 50e-6 s]'
set ylabel 'Vm (V)'

# Flash plot for 5 seconds
plot \
	'axon.out0' with line title 'Soma', \
	'axon.outx' with line title 'Last compartment '

pause 5

# Write images to disk
set term png
set output 'myelin.png'
replot
set output
set term x11
