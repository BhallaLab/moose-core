# Gnuplot script
# First run 'moose Axon.g' to generate *.plot files

set datafile comments '/#'
set title 'Signal propagation along axon'
set xlabel 'Step # [dt = 50e-6 s]'    # This is the plot dt
set ylabel 'Vm (V)'

plot \
	'axon0_0.genesis.plot' every ::1 with line title 'Axon 0 Compartment 0 (Genesis)', \
	'axon100_0.genesis.plot' every ::1 with line title 'Axon 0 Compartment 100 (Genesis)', \
	'axon0_1.genesis.plot' every ::1 with line title 'Axon 1 Compartment 0 (Genesis)', \
	'axon100_1.genesis.plot' every ::1 with line title 'Axon 1 Compartment 100 (Genesis)', \
	'axon0_0.moose.plot' with line title 'Axon 0 Compartment 0 (Moose)', \
	'axon100_0.moose.plot' with line title 'Axon 0 Compartment 100 (Moose)', \
	'axon0_1.moose.plot' with line title 'Axon 1 Compartment 0 (Moose)', \
	'axon100_1.moose.plot' with line title 'Axon 1 Compartment 100 (Moose)'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'axon.png'
replot
set output
set term pop

print "Plot image written to axon.png.\n"
