# Gnuplot script
# First run 'moose Rallpack3.g' to generate *.plot files

set datafile comments '/#'
set title 'Rallpack 3 (Linear cable, with Na and K_DR channels)'
set xlabel 'Step # [dt = 50e-6 s]'
set ylabel 'Vm (V)'

#
# First compartment
#
plot \
	'axon-0.moose.plot' with line title 'Compt 1 (MOOSE)', \
	'axon-0.genesis.plot' every ::1 with line title 'Compt 1 (GENESIS)'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'axon-0.png'
replot
set output
set term pop

print "Plot image written to axon-0.png.\n"

#
# Last compartment
#
plot \
	'axon-x.moose.plot' with line title 'Compt 1000 (MOOSE)', \
	'axon-x.genesis.plot' every ::1 with line title 'Compt 1000 (GENESIS)'

pause mouse key "Any key to continue.\n"

# Write images to disk
set term png
set output 'axon-x.png'
replot
set output
set term pop

print "Plot image written to axon-x.png.\n"
