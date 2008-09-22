# Gnuplot script
# First run 'moose Myelin.g' to generate *.plot files

set datafile comments '/#'
set title 'Myelinated Axon'
set xlabel 'Step # [dt = 100e-6 s]'    # This is the plot dt
set ylabel 'Vm (V)'

plot \
	'axon-0.moose.plot' with line title 'Soma (MOOSE)', \
	'axon-x.moose.plot' with line title 'Last compartment (MOOSE)', \
	'axon-0.genesis.plot' with line title 'Soma (GENESIS)', \
	'axon-x.genesis.plot' with line title 'Last compartment (GENESIS)'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'myelin.png'
replot
set output
set term x11

print "Plot image written to myelin.png.\n"
