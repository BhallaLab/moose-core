# Gnuplot script
# First run 'moose Myelin.g' to generate *.plot files

set datafile comments '/#'
set title 'Myelinated Axon'
set xlabel 'Step # [dt = 50e-6 s]'
set ylabel 'Vm (V)'

# Flash plot for 5 seconds
plot \
	'axon.0.plot' with line title 'Soma', \
	'axon.x.plot' with line title 'Last compartment '

pause .5

# Write images to disk
set term png
set output 'myelin.png'
replot
set output
set term x11
