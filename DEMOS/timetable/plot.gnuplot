# Gnuplot script
# First run 'moose TimeTable.g' to generate *.plot files

set datafile commentschars '/#'
set xlabel 'Step # [dt = 50e-6 s]'    # This is the plot dt

#
# Membrane potential
#
set title 'Synaptic input using TimeTable object: Membrane potential.'
set ylabel 'Vm (V)'
plot \
	'output/Vm.moose.plot' with line title 'MOOSE', \
	'output/Vm.genesis.plot' every ::0::1199 with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'output/Vm.png'
replot
set output
set term wxt

print "Plot image written to output/Vm.png.\n"

#
# Channel conductance
#
set title 'Synaptic input using TimeTable object: Conductance of synaptic channel.'
set ylabel 'Gk (1 / ohm)'
plot \
	'output/Gk.moose.plot' with line title 'MOOSE', \
	'output/Gk.genesis.plot' every ::0::1199 with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'output/Gk.png'
replot
set output
set term wxt

print "Plot image written to output/Gk.png.\n"
