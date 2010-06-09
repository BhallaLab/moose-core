# Gnuplot script
# First run 'moose Axon.g' to generate *.plot files

set datafile comments '/#'
set title 'Signal propagation along axon'
set xlabel 'Step # [dt = 50e-6 s]'    # This is the plot dt
set ylabel 'Vm (V)'

plot \
	'axon0.genesis.plot' every ::1 with line title 'Compartment 0 (Genesis)', \
	'axon100.genesis.plot' every ::1 with line title 'Compartment 100 (Genesis)', \
	'axon200.genesis.plot' every ::1 with line title 'Compartment 200 (Genesis)', \
	'axon300.genesis.plot' every ::1 with line title 'Compartment 300 (Genesis)', \
	'axon400.genesis.plot' every ::1 with line title 'Compartment 400 (Genesis)', \
	'axon0.moose.plot' with line title 'Compartment 0 (Moose)', \
	'axon100.moose.plot' with line title 'Compartment 100 (Moose)', \
	'axon200.moose.plot' with line title 'Compartment 200 (Moose)', \
	'axon300.moose.plot' with line title 'Compartment 300 (Moose)', \
	'axon400.moose.plot' with line title 'Compartment 400 (Moose)'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'axon.png'
replot
set output
set term pop

print "Plot image written to axon.png.\n"
