# Gnuplot script
# First run 'moose mitral.g' to generate *.plot files

set datafile commentschars '/#'
set title 'Mitral cell model.'
set xlabel 'Step # [dt = 50e-6 s]'
set ylabel 'Vm (V)'

# Flash plot for 5 seconds
plot \
	'test.plot' with line title 'MOOSE', \
	'reference.plot' with line title 'GENESIS'

pause 5

# Write images to disk
set term png
set output 'mitral.png'
replot
set output
set term pop
