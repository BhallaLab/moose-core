# Gnuplot script
# First run 'moose network.g' to generate *.plot files

set datafile commentschars '/#'
set title '10x10 network of neurons'
set xlabel 'Step # [dt = 50e-6 s]'
set ylabel 'Vm (V)'

# Flash plot for 5 seconds
#plot \
#	'test.plot' with line title 'MOOSE', \
#	'reference.plot' with line title 'GENESIS'
plot \
	'test0.plot' with line, \
	'test1.plot' with line, \
	'test2.plot' with line, \
	'test3.plot' with line, \
	'test4.plot' with line, \
	'test5.plot' with line, \
	'test6.plot' with line, \
	'test8.plot' with line, \
	'test9.plot' with line

pause .5

# Write images to disk
set term png
set output 'network.png'
replot
set output
set term x11
