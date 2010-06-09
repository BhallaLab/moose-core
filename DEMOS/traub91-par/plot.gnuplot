# Gnuplot script
# First run 'moose Traub91.g' to generate *.plot files

set datafile commentschars '/#'
set title 'Traub CA3 cell model.'
set xlabel 'Step # [dt = 100e-6 s]'    # This is the plot-dt

#
# Membrane potential
#
set title 'Traub CA3 cell model: Membrane potential of soma.'
set ylabel 'Vm (V)'
plot \
	'Vm.moose.plot' with line title 'MOOSE', \
	'Vm.genesis.plot' with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'Vm.png'
replot
set output
set term pop

print "Plot image written to Vm.png.\n"

#
# Calcium concentration
#
set title 'Traub CA3 cell model: [Ca++] in soma.'
set ylabel 'Ca (uM)'
plot \
	'Ca.moose.plot' with line title 'MOOSE', \
	'Ca.genesis.plot' with line title 'GENESIS'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'Ca.png'
replot
set output
set term pop

print "Plot image written to Ca.png.\n"

