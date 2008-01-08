# Gnuplot script
# First run 'moose axon.g' to generate *.plot files

set title 'Signal propagation along axon'
set xlabel 'Step # [dt = 50e-6 s]'
set ylabel 'Vm (V)'

# Flash plot for 5 seconds
plot \
	'axon0.plot' with line title 'Compartment 0', \
	'axon1.plot' with line title 'Compartment 100', \
	'axon2.plot' with line title 'Compartment 200', \
	'axon3.plot' with line title 'Compartment 300', \
	'axon4.plot' with line title 'Compartment 400'

pause 5

# Write images to disk
set term png
set output 'axon.png'
replot
set output
set term x11
